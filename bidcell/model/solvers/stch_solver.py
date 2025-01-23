import torch

class STCHSolver:
    @staticmethod
    def apply(grads, mu=1.0, warmup_epoch=4, current_epoch=0):
        """
        Applies true Tchebycheff scalarization to align gradients, with tasks as the last dimension.

        Args:
            grads (torch.Tensor): A 3D tensor of gradients with shape (num_params, num_batches, num_tasks).
            mu (float): Smooth Tchebycheff coefficient (controls the weight smoothing).
            warmup_epoch (int): Number of epochs for warm-up.
            current_epoch (int): Current training epoch (used for warm-up scheduling).

        Returns:
            torch.Tensor: Aligned gradients.
            torch.Tensor: Computed weights for tasks.
            torch.Tensor: Tchebycheff scalarization values for tasks.
        """
        assert len(grads.shape) == 3, f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable."

        num_params, num_batches, num_tasks = grads.shape

        with torch.no_grad():
            # Transpose tasks to the first dimension for consistent processing
            grads_t = grads.permute(2, 0, 1)  # Shape: (num_tasks, num_params, num_batches)

            # Compute gradient norms across parameters for each task
            grad_norms = torch.norm(grads_t, dim=1)  # Shape: (num_tasks, num_batches)
            grad_norms = torch.clamp(grad_norms, min=1e-6)  # Prevent zero norms

            # Compute max norms across tasks for each batch
            max_norms = grad_norms.max(dim=0).values  # Shape: (num_batches,)
            max_norms = torch.clamp(max_norms, min=1e-6)  # Ensure no division by zero

            # Compute normalized gradient norms
            weighted_norms = grad_norms / max_norms.unsqueeze(0)  # Shape: (num_tasks, num_batches)

            # Apply smoothing with mu
            smoothed_norms = torch.pow(weighted_norms, mu)  # Shape: (num_tasks, num_batches)

            # Normalize weights to sum to 1 across tasks for each batch
            weights = smoothed_norms / smoothed_norms.sum(dim=0, keepdim=True)  # Shape: (num_tasks, num_batches)

            # Warm-up phase: blend with uniform weights
            if current_epoch < warmup_epoch:
                warmup_factor = current_epoch / warmup_epoch
                uniform_weights = torch.ones_like(weights) / num_tasks
                weights = (1.0 - warmup_factor) * uniform_weights + warmup_factor * weights

            # Debugging outputs
            print(f"--- Debugging STCHSolver ---")
            print(f"Shape of grads: {grads.shape}")
            print(f"Shape of grads (after permute): {grads_t.shape}")
            print(f"Gradient Norms (after permute): {grad_norms} (shape={grad_norms.shape})")
            print(f"Max Norms: {max_norms} (shape={max_norms.shape})")
            print(f"Weighted Norms: {weighted_norms} (shape={weighted_norms.shape})")
            print(f"Smoothed Norms (mu={mu}): {smoothed_norms} (shape={smoothed_norms.shape})")
            print(f"Task Weights (before warm-up): {weights} (shape={weights.shape})")
            if current_epoch < warmup_epoch:
                print(f"Warm-up Factor: {warmup_factor}")
                print(f"Task Weights (after warm-up): {weights} (shape={weights.shape})")
            print(f"--- End Debugging ---")

            # Align gradients by scaling with computed weights
            aligned_grads_t = grads_t * weights.unsqueeze(1)  # Shape: (num_tasks, num_params, num_batches)

            # Transpose back to original layout
            aligned_grads = aligned_grads_t.permute(1, 2, 0)  # Shape: (num_params, num_batches, num_tasks)

        return aligned_grads, weights, smoothed_norms
