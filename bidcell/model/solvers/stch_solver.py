import torch

class STCHSolver:
    @staticmethod
    def apply(grads, mu=1.0, warmup_epoch=4, current_epoch=0):
        """
        Applies the STCH-MTL method to process gradients.

        Args:
            grads (torch.Tensor): A 3D tensor of gradients with shape (num_tasks, num_params, num_batches).
            mu (float): Smooth Tchebycheff coefficient (controls the weight smoothing).
            warmup_epoch (int): Number of epochs for warm-up.
            current_epoch (int): Current training epoch (used for warm-up scheduling).

        Returns:
            torch.Tensor: Aligned gradients.
            torch.Tensor: Computed weights for tasks.
            torch.Tensor: Tchebycheff scalarization values for tasks.
        """
        # Ensure the input 'grads' is a 3D tensor
        assert (
            len(grads.shape) == 3
        ), f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable."

        num_tasks, num_params, num_batches = grads.shape

        # Compute the L2 norm of the gradients for each task
        with torch.no_grad():
            grad_norms = torch.norm(grads, dim=1)  # Shape: (num_tasks, num_batches)
            grad_norms = torch.clamp(grad_norms, min=1e-6)  # Prevent zero norms
            
            # Compute the maximum and weighted gradient norms (Tchebycheff scalarization)
            max_norms = grad_norms.max(dim=0).values  # Shape: (num_batches,)
            max_norms = torch.clamp(max_norms, min=1e-6)  # Ensure max norms are not zero
            
            weighted_norms = grad_norms / (max_norms.unsqueeze(0) + 1e-6)  # Shape: (num_tasks, num_batches)

            # Apply smoothing with mu
            smoothed_norms = torch.pow(weighted_norms, mu)  # Shape: (num_tasks, num_batches)

            # Normalize to ensure weights sum to 1 across tasks for each batch
            weights = smoothed_norms / smoothed_norms.sum(dim=0, keepdim=True)  # Shape: (num_tasks, num_batches)

            print("Gradient Norms:", grad_norms)
            print("Weighted Norms:", weighted_norms)
            print("Smoothed Norms:", smoothed_norms)
            print("Task Weights:", weights)

            # Warm-up phase: linearly adjust mu from 1.0 to the target value
            if current_epoch < warmup_epoch:
                warmup_factor = current_epoch / warmup_epoch
                weights = (1.0 - warmup_factor) * (1.0 / num_tasks) + warmup_factor * weights

            # Align gradients by scaling with computed weights
            aligned_grads = grads * weights.unsqueeze(1)  # Shape: (num_tasks, num_params, num_batches)

        return aligned_grads, weights, smoothed_norms
