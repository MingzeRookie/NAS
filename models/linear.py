import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Reduce

class MIL_fc(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1,
                 embed_dim=1024):
        super(MIL_fc,self).__init__()
        assert n_classes == 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifier=  nn.Linear(size[1], n_classes)
        self.top_k=top_k

    def forward(self, h, return_features=False):
        h = self.fc(h)
        logits  = self.classifier(h) # K x 2
        
        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MaxMeanClass(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1, embed_dim=1024):
        super(MaxMeanClass,self).__init__()
        assert n_classes > 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1
    
    def forward(self, h, return_features=False):       
        B,_,_ = h.shape
        h = self.fc(h)
        logits = self.classifiers(h)
        y_probs = F.softmax(logits, dim = 2)
        m = y_probs.view(B, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(B, -1), (m % self.n_classes).view(B, -1)), dim=1)
        Y_prob = []
        for n in range(B):
            Y_prob.append(y_probs[n,top_indices[n,0]])
        Y_prob = torch.stack(Y_prob,dim=0)

        return Y_prob
    
class Mean(nn.Module):
    def __init__(self, input_size, output_class, hidden_dim=512, dropout=0.):
        super(Mean, self).__init__()
        # self.norm = nn.InstanceNorm1d(input_size)
        self.fc = nn.Sequential(nn.Linear(input_size, hidden_dim),
                                nn.ReLU(),
                                nn.Dropout(dropout))
        self.classifiers = nn.Linear(hidden_dim, output_class)
        self.pooling = Reduce('b n c -> b c', 'mean')

    def forward(self, x):
        h = self.fc(x)
        logits = self.classifiers(h)
        Y_prob = self.pooling(logits)
        return Y_prob