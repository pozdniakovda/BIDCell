import torch

# Class that provides a method to align gradients using Procrustes transformation
class ProcrustesSolver:
    @staticmethod
    def apply(grads, scale_mode='min'):
        # Ensure that the input 'grads' is a 3D tensor
        assert (
            len(grads.shape) == 3
        ), f"Invalid shape of 'grads': {grads.shape}. Only 3D tensors are applicable"

        # No gradient calculation, only operations on tensors (detaches from computation graph)
        with torch.no_grad():
            # Compute the covariance matrix of the gradients
            # grads.permute(0, 2, 1) swaps the last two dimensions for matrix multiplication
            cov_grad_matrix_e = torch.matmul(grads.permute(0, 2, 1), grads)
            # Take the mean across all tasks (first dimension)
            cov_grad_matrix_e = cov_grad_matrix_e.mean(0)

            # Perform eigenvalue decomposition of the covariance matrix
            singulars, basis = torch.linalg.eigh(cov_grad_matrix_e)

            # Compute a tolerance level to determine the effective rank of the matrix
            tol = (
                torch.max(singulars)
                * max(cov_grad_matrix_e.shape[-2:])
                * torch.finfo().eps
            )

            # Determine the rank by counting the number of significant eigenvalues
            rank = sum(singulars > tol)

            # Sort the eigenvalues and corresponding eigenvectors in descending order
            order = torch.argsort(singulars, dim=-1, descending=True)
            singulars, basis = singulars[order][:rank], basis[:, order][:, :rank]

            # Scale the eigenvectors based on the specified scaling mode
            if scale_mode == 'min':
                # Scale by the square root of the smallest eigenvalue
                weights = basis * torch.sqrt(singulars[-1]).view(1, -1)
            elif scale_mode == 'median':
                # Scale by the square root of the median eigenvalue
                weights = basis * torch.sqrt(torch.median(singulars)).view(1, -1)
            elif scale_mode == 'rmse':
                # Scale by the root mean square of the eigenvalues
                weights = basis * torch.sqrt(singulars.mean())

            # Normalize the weights by the square root of the eigenvalues
            weights = weights / torch.sqrt(singulars).view(1, -1)

            # Apply the transformation matrix to align the gradients
            weights = torch.matmul(weights, basis.T)
            grads = torch.matmul(grads, weights.unsqueeze(0))

            # Return the aligned gradients, the transformation matrix, and the eigenvalues
            return grads, weights, singulars
