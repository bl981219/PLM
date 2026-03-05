import torch
import torch.nn as nn
import torch.nn.functional as F

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    The input to this loss is the logits of a model, NOT the softmax scores.
    """
    def __init__(self, n_bins=15):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, t=1.0):
        softmaxes = F.softmax(logits/t, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class CrossEntropyLoss(nn.Module):
    def __init__(self, device, t=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        logit_scale = x / self.t
        return F.cross_entropy(logit_scale, target)

class LogitNormLoss(nn.Module):
    def __init__(self, device, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)
    
class TotalVariationLoss(nn.Module):
    def __init__(self, device, gamma=0.5):
        super(TotalVariationLoss, self).__init__()
        self.device = device
        self.gamma = gamma

    def forward(self, x, target):
        nllloss = nn.NLLLoss()
        g = self.gamma
        loss = -nllloss(-torch.log(g + (1-g)*torch.softmax(x, dim=-1))/(1-g), target)
        return loss
    
class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, device, epsilon=0.1):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.device = device
        self.epsilon = epsilon

    def forward(self, x, target):
        logprobs = nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = (1 - self.epsilon) * nll_loss + self.epsilon * smooth_loss
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, device, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss
