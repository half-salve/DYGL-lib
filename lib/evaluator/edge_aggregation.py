import  torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def link_prediction_metric(pos_prob,neg_prob,size=None):
    assert len(pos_prob) == len(neg_prob)
    if size is None:
        size = len(pos_prob)
    
    pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
    pred_label = pred_score > 0.5
    true_label = np.concatenate([np.ones(size), np.zeros(size)])

    acc = (pred_label == true_label).mean()
    ap = average_precision_score(true_label, pred_score)
    f1 = f1_score(true_label, pred_label)
    auc = roc_auc_score(true_label, pred_score)
    return acc,ap,f1,auc



class MergeLayer(torch.nn.Module):#torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)
    def __init__(self, dim1, dim2, dim3, dim4, non_linear=True):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

        # special linear layer for motif explainability
        self.non_linear = non_linear
        if not non_linear:
            assert(dim1 == dim2)
            self.fc = nn.Linear(dim1, 1)
            torch.nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, x1, x2):
        z_walk = None
        if self.non_linear:
            x = torch.cat([x1, x2], dim=-1)
            #x = self.layer_norm(x)
            h = self.act(self.fc1(x))
            z = self.fc2(h)
        else: # for explainability
            # x1, x2 shape: [B, M, F]
            x = torch.cat([x1, x2], dim=-2)  # x shape: [B, 2M, F]
            z_walk = self.fc(x).squeeze(-1)  # z_walk shape: [B, 2M]
            z = z_walk.sum(dim=-1, keepdim=True)  # z shape [B, 1]
            
        return z
    
