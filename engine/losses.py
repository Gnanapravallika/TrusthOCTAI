"""Loss functions including CrossEntropyLoss and Evidential Deep Learning (EDL) Dirichlet Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EDLLoss(nn.Module):
    """Evidential Deep Learning (EDL) Loss based on Dirichlet distribution theory (Sensoy et al., NIPS 2018)."""

    def __init__(self, num_classes: int = 4, annealing_epochs: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs

    def kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        beta = torch.ones((1, self.num_classes), dtype=torch.float32, device=alpha.device)
        s_alpha = torch.sum(alpha, dim=1, keepdim=True)
        s_beta = torch.sum(beta, dim=1, keepdim=True)

        ln_alpha = torch.lgamma(s_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        ln_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(s_beta)

        dg_alpha = torch.digamma(alpha)
        dg_s_alpha = torch.digamma(s_alpha)

        kl = ln_alpha + ln_beta + torch.sum((alpha - beta) * (dg_alpha - dg_s_alpha), dim=1, keepdim=True)
        return kl

    def forward(self, alpha: torch.Tensor, target: torch.Tensor, epoch: int = 0) -> torch.Tensor:
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        s = torch.sum(alpha, dim=1, keepdim=True)

        # Expected cross-entropy under Dirichlet distribution
        a = torch.sum(target_one_hot * (torch.digamma(s) - torch.digamma(alpha)), dim=1, keepdim=True)

        # KL annealing penalty
        annealing_coef = min(1.0, float(epoch) / float(self.annealing_epochs))
        alpha_tilde = target_one_hot + (1.0 - target_one_hot) * alpha
        kl_penalty = annealing_coef * self.kl_divergence(alpha_tilde)

        return torch.mean(a + kl_penalty)


def get_loss_function(loss_type: str = "ce", num_classes: int = 4, annealing_epochs: int = 10) -> nn.Module:
    """Loss function factory."""
    if loss_type.lower() == "edl":
        return EDLLoss(num_classes=num_classes, annealing_epochs=annealing_epochs)
    elif loss_type.lower() == "ce" or loss_type.lower() == "softmax":
        return nn.CrossEntropyLoss()
    else:
        return nn.CrossEntropyLoss()
