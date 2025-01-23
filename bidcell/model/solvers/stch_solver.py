import torch

class STCHSolver:
    @staticmethod
    def apply(grads, mu=1.0, warmup_epoch=4, current_epoch=0, verbose=True):
        """
        Applies true Tchebycheff scalarization to align gradients, using proportional weights.

        Args:
            grads (torch.Tensor): A 3D tensor of gradients with shape (batch_size, num_params, num_tasks).
            mu (float): Smooth Tchebycheff coefficient (controls the weight smoothing).
            warmup_epoch (int): Number of epochs for warm-up.
            current_epoch (int): Current training epoch (used for warm-up scheduling).
            verbose (bool): Whether to print debug information.

        Returns:
            torch.Tensor: Aligned gradients (same shape as input).
            torch.Tensor: Computed weights for tasks.
            torch.Tensor: Tchebycheff scalarization values for tasks.
        """
        assert len(grads.shape) == 3, f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable."

        batch_size, num_params, num_tasks = grads.shape

        with torch.no_grad():
            if verbose:
                print(f"--- Debugging STCHSolver ---")
                print(f"\tShape of grads: {grads.shape}")

            # Compute gradient norms across parameters for each task
            grad_norms = torch.norm(grads, dim=1)  # Shape: (batch_size, num_tasks)
            grad_norms = torch.clamp(grad_norms, min=1e-6)  # Avoid zero norms
            if verbose:
                print(f"\tGradient Norms: {grad_norms} (shape={grad_norms.shape})")

            # Apply smoothing to gradient norms (Tchebycheff scalarization)
            smoothed_norms = torch.pow(grad_norms, mu)  # Shape: (batch_size, num_tasks)
            if verbose:
                print(f"\tSmoothed Norms (mu={mu}): {smoothed_norms} (shape={smoothed_norms.shape})")

            # Compute proportional weights for all tasks
            weights = smoothed_norms / smoothed_norms.sum(dim=1, keepdim=True)  # Shape: (batch_size, num_tasks)
            if verbose:
                print(f"\tTask Weights (before warm-up): {weights} (shape={weights.shape})")

            # Warm-up phase: blend with uniform weights
            if current_epoch < warmup_epoch:
                warmup_factor = current_epoch / warmup_epoch
                uniform_weights = torch.ones_like(weights) / num_tasks
                weights = (1.0 - warmup_factor) * uniform_weights + warmup_factor * weights

            if current_epoch < warmup_epoch:
                if verbose:
                    print(f"\tWarm-up Factor: {warmup_factor}")
                    print(f"\tTask Weights (after warm-up): {weights} (shape={weights.shape})")
            if verbose:
                print(f"--- End Debugging ---")

            # Align gradients by scaling with computed weights
            aligned_grads = grads * weights.unsqueeze(1)  # Shape: (batch_size, num_params, num_tasks)

        return aligned_grads, weights, smoothed_norms
