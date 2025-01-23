import torch
from .solver_utils import (
    to_scalar, 
    track_loss, 
    track_losses, 
    filter_non_contributing, 
    filter_unnecessary,
)
from .procrustes_solver import ProcrustesSolver
from .stch_solver import STCHSolver
from ...config import load_config, Config

def default_solver(optimizer, tracked_losses, loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_mu = None, 
                   loss_pn = None, loss_ne_ov = None, loss_os_ov = None, loss_cc_pn = None, non_contributing_losses=()):
    loss_ne = loss_ne.squeeze() if loss_ne is not None else None
    loss_os = loss_os.squeeze() if loss_os is not None else None
    loss_cc = loss_cc.squeeze() if loss_cc is not None else None
    loss_ov = loss_ov.squeeze() if loss_ov is not None else None
    loss_mu = loss_mu.squeeze() if loss_mu is not None else None
    loss_pn = loss_pn.squeeze() if loss_pn is not None else None
    
    loss_ne_ov = loss_ne_ov.squeeze() if loss_ne_ov is not None else None
    loss_os_ov = loss_os_ov.squeeze() if loss_os_ov is not None else None
    loss_cc_pn = loss_cc_pn.squeeze() if loss_cc_pn is not None else None

    # Sum the contributing losses
    args = filter_non_contributing(loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, 
                                   loss_ne_ov, loss_os_ov, loss_cc_pn, 
                                   non_contributing_losses, assign_none=False)
    contributing_terms, blank_terms, spectator_terms = args
    contributing_terms, unnecessary_terms = filter_unnecessary(contributing_terms)

    loss = sum(list(contributing_terms.values()))

    # Optimisation
    loss.backward()
    optimizer.step()

    # Track individual losses
    keys = ["ne", "os", "cc", "ov", "mu", "pn", "ne_ov", "os_ov", "cc_pn"]
    detached_losses = {}
    for key in keys:
        if contributing_terms.get(key) is not None:
            step_term_loss = contributing_terms[key].detach().cpu().numpy()
        elif spectator_terms.get(key) is not None:
            step_term_loss = spectator_terms[key].detach().cpu().numpy()
        elif unnecessary_terms.get(key) is not None:
            step_term_loss = unnecessary_terms[key].detach().cpu().numpy()
        elif blank_terms.get(key) is not None:
            step_term_loss = blank_terms[key].detach().cpu().numpy()
        else:
            step_term_loss = 0
        detached_losses[key] = step_term_loss

    step_train_loss = loss.detach().cpu().numpy()

    track_losses(tracked_losses = tracked_losses, 
                 loss_ne = detached_losses["ne"], 
                 loss_os = detached_losses["os"], 
                 loss_cc = detached_losses["cc"], 
                 loss_ov = detached_losses["ov"], 
                 loss_mu = detached_losses["mu"], 
                 loss_pn = detached_losses["pn"], 
                 loss_ne_ov = detached_losses["ne_ov"], 
                 loss_os_ov = detached_losses["os_ov"], 
                 loss_cc_pn = detached_losses["cc_pn"], 
                 loss_total = step_train_loss)

    return step_train_loss

def procrustes_method(model, optimizer, tracked_losses, loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_mu = None, 
                      loss_pn = None, loss_ne_ov = None, loss_os_ov = None, loss_cc_pn = None, scale_mode = "min", non_contributing_losses=()): 
    # Remove non-contributing losses
    args = filter_non_contributing(loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, 
                                   loss_ne_ov, loss_os_ov, loss_cc_pn, 
                                   non_contributing_losses, assign_none=False)
    contributing_terms, blank_terms, spectator_terms = args
    contributing_terms, unnecessary_terms = filter_unnecessary(contributing_terms)
                          
    # Backward pass
    grads = []
    for key, loss in contributing_terms.items():
        optimizer.zero_grad()  # Clear previous gradients
        try:
            loss.backward(retain_graph=True)  # Retain graph for backpropagation
        except Exception as e:
            raise Exception(f"Contributing loss {key} of type {type(loss)} produced the following exception during backpropagation: \n\t{e}")
        grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in model.parameters()])
        grads.append(grad)

    grads = torch.stack(grads, dim=0)  # Stack gradients

    # Perform backward pass on spectator losses
    for loss in spectator_terms.values():
        optimizer.zero_grad()
        try:
            loss.backward(retain_graph=True)
        except Exception as e:
            raise Exception(f"Spectator loss [{key}] of type [{type(loss)}] produced the following exception during backpropagation: \n\t{e}")


    # Apply Procrustes Solver
    grads, weights, singulars = ProcrustesSolver.apply(grads.T.unsqueeze(0), scale_mode)
    grad, weights = grads[0].sum(-1), weights.sum(-1)

    # Apply aligned gradients to model parameters
    offset = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        _offset = offset + p.grad.shape.numel()
        p.grad.data = grad[offset:_offset].view_as(p.grad)
        offset = _offset

    # Perform optimization step
    optimizer.step()

    # Calculate total loss with Procrustes-processed losses
    total_loss = sum(list(contributing_terms.values()))

    # Track the loss values for graphing purposes
    keys = ["ne", "os", "cc", "ov", "mu", "pn", "ne_ov", "os_ov", "cc_pn"]
    scalarized_losses = {}
    for key in keys:
        if contributing_terms.get(key) is not None:
            scalar_loss = to_scalar(contributing_terms[key])
        elif spectator_terms.get(key) is not None:
            scalar_loss = to_scalar(spectator_terms[key])
        elif unnecessary_terms.get(key) is not None:
            scalar_loss = to_scalar(unnecessary_terms[key])
        elif blank_terms.get(key) is not None:
            scalar_loss = to_scalar(blank_terms[key])
        else:
            scalar_loss = 0
        scalarized_losses[key] = scalar_loss

    total_loss_scalar = to_scalar(total_loss)
    
    track_losses(tracked_losses = tracked_losses, 
                 loss_ne = scalarized_losses.get("ne"), 
                 loss_os = scalarized_losses.get("os"), 
                 loss_cc = scalarized_losses.get("cc"), 
                 loss_ov = scalarized_losses.get("ov"), 
                 loss_mu = scalarized_losses.get("mu"), 
                 loss_pn = scalarized_losses.get("pn"), 
                 loss_ne_ov = scalarized_losses.get("ne_ov"), 
                 loss_os_ov = scalarized_losses.get("os_ov"), 
                 loss_cc_pn = scalarized_losses.get("cc_pn"), 
                 loss_total = total_loss_scalar)

    return total_loss_scalar

def stch_method(model, optimizer, tracked_losses, loss_ne=None, loss_os=None, loss_cc=None, loss_ov=None, loss_mu=None,
                loss_pn=None, loss_ne_ov=None, loss_os_ov=None, loss_cc_pn=None, mu=1.0, warmup_epoch=4, current_epoch=0,
                non_contributing_losses=(), verbose=False):
    """
    Applies STCH-MTL to align gradients dynamically using smooth Tchebycheff scalarization. 
    
    Based on the ICML2024 paper by Lin et al.: "Smooth Tchebycheff Scalarization for Multi-Objective Optimization"
    Link to their GitHub: https://github.com/Xi-L/STCH/tree/main

    Args:
        model:                   The PyTorch model
        optimizer:               The optimizer used for gradient updates
        tracked_losses:          A dictionary to track the scalarized loss values
        loss_*:                  Individual loss terms (e.g., loss_ne, loss_os, etc.)
        mu:                      Smooth Tchebycheff coefficient for task weight smoothing
        warmup_epoch:            Number of warm-up epochs to stabilize weights
        current_epoch:           The current training epoch (for warm-up adjustment)
        non_contributing_losses: A list of non-contributing loss keys to exclude from scalarization
        verbose:                 Whether to provide extensive information on tensor shapes and contents. 

    Returns:
        float: The total scalarized loss value for tracking.
    """
    # Remove non-contributing losses
    args = filter_non_contributing(
        loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn,
        loss_ne_ov, loss_os_ov, loss_cc_pn, non_contributing_losses, assign_none=False
    )
    contributing_terms, blank_terms, spectator_terms = args
    contributing_terms, unnecessary_terms = filter_unnecessary(contributing_terms)

    # Collect gradients for contributing losses
    grads = []
    for key, loss in contributing_terms.items():
        optimizer.zero_grad()  # Clear previous gradients
        try:
            loss.backward(retain_graph=True)  # Retain computation graph for multiple losses
        except Exception as e:
            raise Exception(f"Contributing loss {key} produced an exception during backpropagation: \n{e}")
        grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in model.parameters()])
        grads.append(grad)

    grads = torch.stack(grads, dim=0)  # Stack gradients

    # Perform backward pass on spectator losses
    for loss in spectator_terms.values():
        optimizer.zero_grad()
        try:
            loss.backward(retain_graph=True)
        except Exception as e:
            raise Exception(f"Spectator loss produced an exception during backpropagation: \n{e}")

    # Apply STCHSolver to align gradients
    aligned_grads, weights, tchebycheff_values = STCHSolver.apply(
        grads.T.unsqueeze(0), mu=mu, warmup_epoch=warmup_epoch, current_epoch=current_epoch, verbose=verbose
    )
    aligned_grads = aligned_grads[0].sum(-1)  # Aggregate aligned gradients

    # Assign aligned gradients back to model parameters
    offset = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        _offset = offset + p.grad.shape.numel()
        p.grad.data = aligned_grads[offset:_offset].view_as(p.grad)
        offset = _offset

    # Perform optimization step
    optimizer.step()

    # Calculate total loss with STCH-processed gradients
    total_loss = sum(list(contributing_terms.values()))

    # Track individual loss values for monitoring
    keys = ["ne", "os", "cc", "ov", "mu", "pn", "ne_ov", "os_ov", "cc_pn"]
    scalarized_losses = {}
    for key in keys:
        if contributing_terms.get(key) is not None:
            scalar_loss = to_scalar(contributing_terms[key])
        elif spectator_terms.get(key) is not None:
            scalar_loss = to_scalar(spectator_terms[key])
        elif unnecessary_terms.get(key) is not None:
            scalar_loss = to_scalar(unnecessary_terms[key])
        elif blank_terms.get(key) is not None:
            scalar_loss = to_scalar(blank_terms[key])
        else:
            scalar_loss = 0
        scalarized_losses[key] = scalar_loss

    total_loss_scalar = to_scalar(total_loss)

    # Track the losses for logging
    track_losses(
        tracked_losses=tracked_losses,
        loss_ne=scalarized_losses.get("ne"),
        loss_os=scalarized_losses.get("os"),
        loss_cc=scalarized_losses.get("cc"),
        loss_ov=scalarized_losses.get("ov"),
        loss_mu=scalarized_losses.get("mu"),
        loss_pn=scalarized_losses.get("pn"),
        loss_ne_ov=scalarized_losses.get("ne_ov"),
        loss_os_ov=scalarized_losses.get("os_ov"),
        loss_cc_pn=scalarized_losses.get("cc_pn"),
        loss_total=total_loss_scalar
    )

    return total_loss_scalar
