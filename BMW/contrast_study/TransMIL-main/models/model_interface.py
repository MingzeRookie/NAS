import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd
import argparse
#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import cross_entropy_torch

#---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

#---->
import pytorch_lightning as pl


class  ModelInterface(pl.LightningModule):

    #---->init
    # def __init__(self, model, loss, optimizer, **kargs):
    #     super(ModelInterface, self).__init__()
    #     self.save_hyperparameters()
    #     self.load_model()
    #     self.loss = create_loss(loss)
    #     self.optimizer = optimizer
    #     self.n_classes = model.n_classes
    #     self.log_path = kargs['log']
    # 在 ModelInterface 类的 __init__ 方法中
    def __init__(self, model, loss, optimizer, **kargs): # model, loss, optimizer, kargs['data'] 都是 addict.Dict 配置对象
        super(ModelInterface, self).__init__()
        def to_plain_dict(cfg_dict):
            if isinstance(cfg_dict, dict): # addict.Dict 是 dict 的子类
                return {k: to_plain_dict(v) for k, v in cfg_dict.items()}
            elif isinstance(cfg_dict, list):
                return [to_plain_dict(i) for i in cfg_dict]
            return cfg_dict

        hparams_to_log = {
            'model': to_plain_dict(model),         # <--- 应用转换
            'loss': to_plain_dict(loss),           # <--- 应用转换
            'optimizer': to_plain_dict(optimizer), # <--- 应用转换
            'data': to_plain_dict(kargs['data']),  # <--- 应用转换
            'log_path_str': str(kargs['log'])      # Path 对象转换为字符串
        }
        self.save_hyperparameters(hparams_to_log)
        self.log_path_internal = kargs['log'] 
        self.load_model() 
        # self.loss = create_loss(self.hparams.loss)
        loss_config_dict = self.hparams.loss 
        if isinstance(loss_config_dict, dict):
            loss_config_for_factory = argparse.Namespace(**loss_config_dict)
        else:
            loss_config_for_factory = loss_config_dict
            
        self.loss = create_loss(loss_config_for_factory)
        self.n_classes = self.hparams.model['n_classes']
        
        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
#---->Metrics using torchmetrics
        # 首先根据 self.n_classes 确定任务类型和用于度量的类别数参数
        if self.n_classes == 1: 
            # 通常用于sigmoid输出的二分类 (单个输出神经元)
            task_type = 'binary'
            num_classes_for_auroc_kappa = self.n_classes # 通常为 1
            num_classes_for_other_metrics = None # 或 1，取决于具体度量API
        elif self.n_classes == 2:
            task_type = 'binary'
            num_classes_for_auroc_kappa = self.n_classes # 为 2
            num_classes_for_other_metrics = None 
        elif self.n_classes > 2:
            task_type = 'multiclass'
            num_classes_for_auroc_kappa = self.n_classes
            num_classes_for_other_metrics = self.n_classes 
        else:
            # n_classes 为0或负数是非法的
            raise ValueError(f"n_classes must be >= 1, but received {self.n_classes}")

        # 初始化 torchmetrics 度量
        try:
            # 对于 torchmetrics 0.7.0 及更高版本，使用 F1Score
            self.F1 = torchmetrics.F1Score(task=task_type, num_classes=num_classes_for_other_metrics, average='macro')
        except AttributeError:
            # 针对可能非常旧的 torchmetrics 版本的回退 (理论上根据 environment.yml 不会触发)
            print("Warning: torchmetrics.F1Score not found, falling back to torchmetrics.F1. Consider updating torchmetrics.")
            self.F1 = torchmetrics.F1(task=task_type, num_classes=num_classes_for_other_metrics, average='macro')
        except TypeError as e_f1:
            # 处理 F1Score/F1 存在但参数不匹配的情况 (例如，对二分类传递了不必要的 num_classes)
            if "unexpected keyword argument 'num_classes'" in str(e_f1) and task_type == 'binary':
                print(f"Adjusting F1Score/F1 for binary task due to: {e_f1}")
                F1Class = getattr(torchmetrics, 'F1Score', torchmetrics.F1) # 动态选择存在的类
                self.F1 = F1Class(task=task_type, average='macro')
            else:
                raise e_f1 # 重新抛出未预料的 TypeError
        
        self.AUROC = torchmetrics.AUROC(task=task_type, num_classes=num_classes_for_auroc_kappa, average='macro')
        self.Accuracy = torchmetrics.Accuracy(task=task_type, num_classes=num_classes_for_other_metrics, average='micro') # 'micro' 通常用于整体准确率
        self.CohenKappa = torchmetrics.CohenKappa(task=task_type, num_classes=num_classes_for_auroc_kappa) # CohenKappa 通常需要 num_classes
        self.Recall = torchmetrics.Recall(task=task_type, num_classes=num_classes_for_other_metrics, average='macro')
        self.Precision = torchmetrics.Precision(task=task_type, num_classes=num_classes_for_other_metrics, average='macro')
        self.Specificity = torchmetrics.Specificity(task=task_type, num_classes=num_classes_for_other_metrics, average='macro')

        # 创建 MetricCollection 用于批量记录
        metrics_list_for_collection = [
            self.Accuracy, 
            self.CohenKappa, 
            self.F1, 
            self.Recall, 
            self.Precision, 
            self.Specificity
            # self.AUROC 通常作为主要监控指标或单独记录，但也可以加入 MetricCollection
        ]
        
        metrics_collection = torchmetrics.MetricCollection(metrics_list_for_collection)
        
        self.valid_metrics = metrics_collection.clone(prefix='val_')
        self.test_metrics = metrics_collection.clone(prefix='test_')

        #--->random
        self.shuffle = self.hparams.data['data_shuffle']
        self.count = 0


    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        #---->inference
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->loss
        loss = self.loss(logits, label)

        #---->acc log
        Y_hat = int(Y_hat)
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        return {'loss': loss} 

    def training_epoch_end(self, training_step_outputs):
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def validation_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']


        #---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}


    def validation_epoch_end(self, val_step_outputs):
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.stack([x['label'] for x in val_step_outputs], dim = 0)
        
        #---->
        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)

        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)
    


    # 在 ModelInterface 类中
    def configure_optimizers(self):
        optimizer_config_dict = self.hparams.optimizer 
        if isinstance(optimizer_config_dict, dict):
            optimizer_config_for_factory = argparse.Namespace(**optimizer_config_dict)
        else:
            optimizer_config_for_factory = optimizer_config_dict
        optimizer = create_optimizer(optimizer_config_for_factory, self.model)
        return [optimizer] # PyTorch Lightning 期望一个包含优化器的列表

    def test_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def test_epoch_end(self, output_results):
        probs = torch.cat([x['Y_prob'] for x in output_results], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        target = torch.stack([x['label'] for x in output_results], dim = 0)
        
        #---->
        auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
        metrics['auc'] = auc
        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()
        print()
        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        #---->
        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path_internal / 'result.csv')


    def load_model(self):
        name = self.hparams.model['name']
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        model_hparams_dict = self.hparams.model
        inkeys = model_hparams_dict.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = model_hparams_dict[arg]
        args1.update(other_args)
        return Model(**args1)