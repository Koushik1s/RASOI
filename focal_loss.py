import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss_my_create(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss_my_create, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probas = torch.sigmoid(inputs)

        pt = targets * probas + (1 - targets) * (1 - probas)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

import sklearn.linear_model as l1
