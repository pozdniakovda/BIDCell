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

    def __init__(self, preference_weights, device, verbose=False) -> None:
        super(STCHLoss, self).__init__()
        self.preference_weights = preference_weights
        self.device = device
        self.verbose = verbose

    def forward(self, losses, preference_weights=None, ideal_vals=None, mu=1.0):
        if preference_weights is None:
            if self.preference_weights is not None:
                preference_weights = self.preference_weights
            else: 
                preference_weights = np.ones(len(losses))

        if ideal_vals is None:
            ideal_vals = np.zeros(len(losses), dtype=float)

        # Convert to torch tensors on device
        if not torch.is_tensor(preference_weights):
            preference_weights = torch.tensor(preference_weights, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(ideal_vals):
            ideal_vals = torch.tensor(ideal_vals, dtype=torch.float32, device=self.device)

        stch_losses = []
        for idx, (preference_weight, loss, ideal_val) in enumerate(zip(preference_weights, losses, ideal_vals)):
            if not torch.is_tensor(loss):
                loss = torch.tensor(loss, dtype=torch.float64, device=self.device)
            elif loss.dtype == torch.float16 or loss.dtype == torch.float32:
                original_dtype = str(loss.dtype)
                loss = loss.to(torch.float64)
                if self.verbose: 
                    print(f"\tNotice: for loss #{idx+1}, loss dtype={original_dtype}, so was recast to {loss.dtype}")
            
            if not torch.is_tensor(ideal_val):
                ideal_val = torch.tensor(ideal_val, dtype=torch.float64, device=self.device)
            if not torch.is_tensor(preference_weight):
                preference_weight = torch.tensor(preference_weight, dtype=torch.float64, device=self.device)

            # Subtract ideal value (usually zero for most objectives)
            stch_loss = loss - ideal_val
            if torch.isinf(stch_loss) or torch.isnan(stch_loss):
                print(f"\tERROR: for loss #{idx+1}, after subtracting ideal_val={ideal_val}, stch_loss is {stch_loss.item()}")

            # Apply the preference weight to the loss
            stch_loss = stch_loss * preference_weight
            if torch.isinf(stch_loss) or torch.isnan(stch_loss):
                print(f"\tERROR: for loss #{idx+1}, after multiplying by preference_weight={preference_weight}, stch_loss is {stch_loss.item()}")

            # Divide by the smoothing factor mu
            stch_loss = stch_loss / mu
            if torch.isinf(stch_loss) or torch.isnan(stch_loss):
                print(f"\tERROR: for loss #{idx+1}, after dividing by mu={mu}, stch_loss is {stch_loss.item()}")

            # Take the exponential
            stch_loss = torch.exp(stch_loss)
            if torch.isinf(stch_loss) or torch.isnan(stch_loss):
                print(f"\tERROR: for loss #{idx+1}, after torch.exp(), stch_loss is {stch_loss.item()}")

            stch_losses.append(stch_loss)

        # Sum the exponentials
        stch_losses = torch.stack(stch_losses)
        stch_loss = torch.sum(stch_losses)
        if torch.isinf(stch_loss) or torch.isnan(stch_loss):
            print(f"ERROR: Sum of STCH-scalarized losses is {stch_loss.item()}")

        # Take the natural logarithm
        stch_loss = torch.log(stch_loss)
        if torch.isinf(stch_loss) or torch.isnan(stch_loss):
            print(f"ERROR: Final STCH loss is {stch_loss.item()}")

        # Rescale by mu
        stch_loss = stch_loss * mu
        if torch.isinf(stch_loss) or torch.isnan(stch_loss):
            print(f"ERROR: Final scaled STCH loss: {stch_loss.item()}")

        return stch_loss


class DBMTLLoss(nn.Module):
    """
    Dual-Balancing Multi-Task Learning (DB-MTL) loss with loss-scale and gradient-magnitude balancing.
    This loss function applies a logarithmic transformation to task losses and normalizes task gradients.
    Based on Lin et al. (2022) "Dual-Balancing Multi-Task Learning"
    """
    
    def __init__(self, preference_weights=None, device="cpu") -> None:
        super(DBMTLLoss, self).__init__()
        self.preference_weights = preference_weights
        self.device = device
    
    def forward(self, losses, model, optimizer, preference_weights=None, epsilon=1e-8):
        """
        Computes the DB-MTL loss with logarithmic transformation and gradient-magnitude balancing.
        
        Args:
            losses (list of tensors): The individual task losses.
            model (torch.nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
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
        log_transformed_losses = []
        for loss, weight in zip(losses, preference_weights):
            loss_adj = loss + epsilon
            log_loss = torch.log(loss_adj)
            weighted_log_loss = log_loss * weight
            log_transformed_losses.append(weighted_log_loss)
        
        # Compute total loss
        log_total_loss = torch.sum(torch.stack(log_transformed_losses))
        total_loss = torch.sum(torch.stack(losses))
        
        return (log_transformed_losses, log_total_loss, total_loss)
