import numpy as np
import torch
import torch.nn as nn


class SummedLoss(nn.Module):
    """
    Performs a simple summation of input losses.
    """

    def __init__(self, device) -> None:
        super(SummedLoss, self).__init__()
        self.device = device

    def forward(self, losses):
        # Simple summation
        losses = torch.stack(losses)
        loss = torch.sum(losses)

        return loss


class STCHLoss(nn.Module):
    """
    Performs smoothed Tchebycheff scalarization (STCH) to sum a set of losses.
    The summed loss can then be subjected to gradient descent. 
    
    Based on Lin et al. (2024) ICML2024 Paper: 
    "Smooth Tchebycheff Scalarization for Multi-Objective Optimization"
    """

    def __init__(self, preference_weights, device) -> None:
        super(STCHLoss, self).__init__()
        self.preference_weights = preference_weights
        self.device = device

    def forward(self, losses, preference_weights=None, ideal_vals=None, mu=1.0):
        # Preference weights are equivalent to loss weights in the config
        if preference_weights is None:
            if self.preference_weights is not None:
                preference_weights = self.preference_weights
            else: 
                preference_weights = np.ones(len(losses))

        # These are used to evaluate the  distance to an ideal loss value
        if ideal_vals is None:
            ideal_vals = np.zeros(len(losses), dtype=float)

        # Create a list of STCH-scalarized losses to sum
        stch_losses = []
        for preference_weight, loss, ideal_val in zip(preference_weights, losses, ideal_vals):
            stch_loss = loss - ideal_val # distance to idea value
            stch_loss = stch_loss * preference_weight # applies the weight for this loss
            stch_loss = stch_loss / mu # divides by a smoothing factor, usually between 0.5 and 2.0
            stch_loss = torch.exp(stch_loss) # takes the exponential
            stch_losses.append(stch_loss)

        # Sum the STCH-scalarized loss terms
        stch_losses = torch.stack(stch_losses)
        stch_loss = torch.sum(stch_losses)

        # Take the logarithm of the summed losses and scale by mu
        stch_loss = torch.log(stch_loss)
        stch_loss = stch_loss * mu

        return stch_loss


class DBMTLLoss(nn.Module):
    """
    Dual-Balancing Multi-Task Learning (DB-MTL) loss with loss-scale balancing.
    This loss function applies a logarithmic transformation to task losses to balance their scales.
    Based on Lin et al. (2022) "Dual-Balancing Multi-Task Learning"
    """
    
    def __init__(self, preference_weights=None, device='cpu') -> None:
        super(DBMTLLoss, self).__init__()
        self.preference_weights = preference_weights
        self.device = device
    
    def forward(self, losses, preference_weights=None, epsilon=1e-8):
        """
        Computes the DB-MTL loss with logarithmic transformation.
        
        Args:
            losses (list of tensors): The individual task losses.
            preference_weights (list or tensor, optional): The task weights. Defaults to equal weights.
            epsilon (float, optional): A small value to prevent log(0). Defaults to 1e-8.
        
        Returns:
            total_loss: The total DB-MTL loss.
        """
        
        # Set preference weights
        if preference_weights is None:
            if self.preference_weights is not None:
                preference_weights = self.preference_weights
            else:
                preference_weights = torch.ones(len(losses), device=self.device)
        
        # Apply the logarithmic transformation to balance loss scales
        log_transformed_losses = [torch.log(loss + epsilon) * weight for loss, weight in zip(losses, preference_weights)]
        
        # Sum the transformed losses
        total_loss = torch.sum(torch.stack(log_transformed_losses))
        
        return total_loss
