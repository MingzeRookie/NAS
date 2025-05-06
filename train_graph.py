import argparse
import logging
import os
import random
import sys
import time
import warnings
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# import models.abmil as abmil
import models
from datasets_graph import cross_validation
# import models.abmil as abmil
# import models.linear as linear
# from models.mambamil import MambaMIL
from models.patchgcn import PatchGCN_Surv
# from utils.ORLoss import OrdinalRegressionLoss

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def multi_label_roc(labels, predictions, num_classes):
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(
            fpr, tpr, threshold
        )
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal


def get_datasets(cfg):
    return cross_validation(cfg)


def get_model(cfg):
    model_cfg = cfg["model"]
    if model_cfg['name'] == 'abmil':
        milnet = abmil.GatedAttention(
            model_cfg.get("feats_dim", 384), cfg['num_class']
        ).cuda()
    elif model_cfg['name'] == 'mean':
        milnet = linear.Mean(model_cfg.get("feats_dim", 384), cfg['num_class']).cuda()
    elif model_cfg['name'] == 'max':
        milnet = linear.MaxMeanClass(embed_dim = model_cfg.get("feats_dim", 384), n_classes = cfg['num_class']).cuda()
    elif model_cfg['name'] == 'mamba':
        milnet = MambaMIL(in_dim=model_cfg.get("feats_dim", 384),n_classes=cfg['num_class'],dropout=0.5,act='gelu').cuda()
    elif model_cfg['name'] == 'graph':
        milnet = PatchGCN_Surv(num_features=model_cfg.get("feats_dim", 384),n_classes=cfg['num_class']).cuda()
    return milnet


def get_optimizer(cfg, model):

    if cfg["num_class"] > 2:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    num_epochs = cfg["optimizer"].get("epoch")
    lr = cfg["optimizer"].get("lr")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 0)

    return num_epochs, criterion, optimizer, scheduler


def train(train_loader, milnet, criterion, optimizer, cfg):
    milnet.train()
    total_loss = 0
    for batch, (bags, bag_label) in enumerate(train_loader, start=1):
        # for i in enumerate(train_loader):
        optimizer.zero_grad()
        bags = bags.cuda()
        bag_label = bag_label.cuda()
        # with torch.autograd.set_detect_anomaly(True):
        bag_prediction = milnet(bags)
        # bag_label = F.one_hot(bag_label.long(), num_classes=cfg["num_class"]).float()
        # print(bag_prediction,bag_label.long())
        loss = criterion(bag_prediction, bag_label.long())
        # print(bags.x)
        loss.backward()
        optimizer.step()
        # print(bag_prediction,bag_label)
        # print(loss.item())
        sys.stdout.write(
            "\r Training batch [%d/%d] batch loss: %.4f"
            % (batch, len(train_loader), loss.item())
        )
        total_loss = total_loss + loss.item()
        del bags, bag_label, bag_prediction, loss
        torch.cuda.empty_cache()
        gc.collect()
    return total_loss / len(train_loader)


def test(test_loader, milnet, criterion, cfg):
    milnet.eval()
    gt = []
    pred_labels = []
    test_predictions = []
    total_loss = 0
    with torch.no_grad():
        for bags, bag_label in test_loader:
            gt.extend(bag_label.int().numpy().tolist())
            bags = bags.cuda()
            bag_prediction = milnet(bags).cpu()
            bag_loss = criterion(bag_prediction, bag_label.long())
            bag_prediction = torch.softmax(bag_prediction.squeeze(), -1).numpy()
            test_predictions.extend([bag_prediction])
            pred_label = np.argmax(bag_prediction)
            pred_labels.append(pred_label)
            total_loss = total_loss + bag_loss.item()
    onehot_gt = np.identity(cfg["num_class"])[gt]
    test_predictions = np.array(test_predictions)
    acc = accuracy_score(gt, pred_labels)
    precision = precision_score(gt, pred_labels, average="macro")
    recall = recall_score(gt, pred_labels, average="macro")
    auc = roc_auc_score(onehot_gt, test_predictions, average="macro", multi_class="ovo")
    f1_score = 2 * precision * recall / (precision + recall)
    kappa = cohen_kappa_score(gt, pred_labels, weights="quadratic")
    return total_loss / len(test_loader), acc, precision, recall, auc, f1_score, kappa


def main(args):
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    setup_seed(config.get("random_seed", 7777))
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())
    save_path = config.get("output", "runs")
    os.makedirs(save_path, exist_ok=True)
    # with open(os.path.join(save_path, "config.yaml"), "w") as f:
    #     yaml.dump(config, f)

    kfold_dataset = get_datasets(config)

    for fold, (train_loader, test_loader) in enumerate(kfold_dataset):
        best_acc, best_precision, best_recall, best_auc = 0, 0, 0, 0
        milnet = get_model(config)
        num_epochs, criterion, optimizer, scheduler = get_optimizer(config, milnet)
        print(f"Training on fold {fold + 1}")
        for epoch in range(1, num_epochs):
            train_loss = train(
                train_loader, milnet, criterion, optimizer, config
            )  # iterate all bags
            # torch.cuda.empty_cache()
            test_loss, test_acc, precision, recall, auc, f1_score, kappa = test(
                test_loader, milnet, criterion, config
            )
            print(
                f"""\nEpoch [{epoch}/{num_epochs}] [Train]- loss: {train_loss}
                [Test]--- LOSS : {test_loss}, ACC: {test_acc}, AUC:{auc}, Kappa: {kappa}"""
            )
            scheduler.step()
            if test_acc > best_acc:
                best_acc = test_acc
                best_precision = precision
                best_recall = recall
                best_auc = np.mean(auc)
                best_f1 = f1_score
                best_kappa = kappa
                # torch.save(milnet, os.path.join(save_path, "best_model.pth"))
                # print(f"Best model saved at: {save_path}")
        with open(os.path.join(save_path, "log_metric.txt"), "a+") as f:
            f.write(
                f"FOLD: {fold}: [Test]- ACC:{best_acc}, AUC:{best_auc}, Kappa:{best_kappa},F1:{best_f1}, Precision:{best_precision}, Recall:{best_recall}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="config.yaml")
    # parser.add_argument("--local-rank", type=int, default=-1)
    # parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--port", default=None, type=int)
    args = parser.parse_args()
    main(args)
