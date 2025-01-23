import torch

class STCHSolver:
    @staticmethod
    def apply(grads, mu=1.0, warmup_epoch=4, current_epoch=0):
        """
        Applies true Tchebycheff scalarization to align gradients, with debug outputs.

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
        assert len(grads.shape) == 3, f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable."

        num_tasks, num_params, num_batches = grads.shape

        with torch.no_grad():
            # Compute gradient norms
            grad_norms = torch.norm(grads, dim=1)  # Shape: (num_tasks, num_batches)
            grad_norms = torch.clamp(grad_norms, min=1e-6)  # Avoid zero norms

            # Apply smoothing to gradients (Tchebycheff scalarization)
            smoothed_norms = torch.pow(grad_norms, mu)  # Shape: (num_tasks, num_batches)

            # Identify the worst-performing task for each batch
            tchebycheff_values, worst_task_indices = smoothed_norms.max(dim=0)  # Shape: (num_batches,)

            # Convert to weights (emphasize the worst task)
            weights = torch.zeros_like(smoothed_norms)  # Initialize weights
            weights[worst_task_indices, range(num_batches)] = 1.0  # Assign full weight to the worst task

            # Warm-up phase: blend with uniform weights
            if current_epoch < warmup_epoch:
                warmup_factor = current_epoch / warmup_epoch
                uniform_weights = torch.ones_like(weights) / num_tasks
                weights = (1.0 - warmup_factor) * uniform_weights + warmup_factor * weights

            # Debugging outputs
            print(f"--- Debugging STCHSolver ---")
            print(f"Shape of grads: {grads.shape}")
            print(f"Gradient Norms: {grad_norms} (shape={grad_norms.shape})")
            print(f"Smoothed Norms (mu={mu}): {smoothed_norms} (shape={smoothed_norms.shape})")
            print(f"Tchebycheff Values: {tchebycheff_values} (shape={tchebycheff_values.shape})")
            print(f"Worst Task Indices: {worst_task_indices} (shape={worst_task_indices.shape})")
            print(f"Task Weights (before warm-up): {weights} (shape={weights.shape})")
            if current_epoch < warmup_epoch:
                print(f"Warm-up Factor: {warmup_factor}")
                print(f"Task Weights (after warm-up): {weights} (shape={weights.shape})")
            print(f"--- End Debugging ---")

            # Align gradients by scaling with computed weights
            aligned_grads = grads * weights.unsqueeze(1)  # Shape: (num_tasks, num_params, num_batches)

        return aligned_grads, weights, tchebycheff_values

