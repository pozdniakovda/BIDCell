import torch
from .solver_utils import (
    to_scalar, 
    track_loss, 
    track_losses, 
    filter_losses,
)
from .procrustes_solver import ProcrustesSolver
from ..model.loss_summation import SummedLoss, STCHLoss, DBMTLLoss
from ...config import load_config, Config

loss_key_conversion = {"ne": "Nuclei Encapsulation Loss", 
                       "os": "Oversegmentation Loss", 
                       "cc": "Cell Calling Loss", 
                       "ov": "Overlap Loss", 
                       "mu": "Multiple Assignment Loss", 
                       "pn": "Pos-Neg Marker Loss", 
                       "ne_ov": "Combined Nuclei Encapsulation and Overlap Loss", 
                       "os_ov": "Combined Oversegmentation and Overlap Loss", 
                       "cc_pn": "Combined Cell Calling and Marker Loss", 
                       "total": "Total Loss"}

def summed_solver(optimizer, device, tracked_losses, model = None, 
                  loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_mu = None, loss_pn = None, 
                  loss_ne_ov = None, loss_os_ov = None, loss_cc_pn = None, non_contributing_losses=(), 
                  sum_mode = "arithmetic", preference_weights = None, ideal_vals = None, stch_mu = 1.0, dbmtl_epsilon = None):
    # Default solver for summed losses

    # Filter the losses based on whether they contribute to the summed loss
    filtered_losses = filter_losses(optimizer, loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, 
                                    loss_ne_ov, loss_os_ov, loss_cc_pn, non_contributing_losses, squeeze=True)
    contributing_losses, unnecessary_losses, blank_losses, spectator_losses = filtered_losses

    # Sum the contributing losses
    if sum_mode in ["stch", "smooth_tchebycheff", "smoothed_tchebycheff"]:
        criterion_stch = STCHLoss(preference_weights, device)
        loss = criterion_stch(list(contributing_losses.values()), preference_weights, ideal_vals, stch_mu)
        # Optimisation
        loss.backward()
        optimizer.step()
        
    elif sum_mode in ["dbmtl", "db-mtl"]:
        print(f"preference_weights: {preference_weights}")
        print(f"Constructing DBMTLLoss...")
        criterion_dbmtl = DBMTLLoss(preference_weights, device)
        if dbmtl_epsilon is None:
            raise Exception(f"sum_mode was set to {sum_mode}, but dbmtl_epsilon was not given (None)")
        if model is None:
            raise Exception(f"sum_mode was set to {sum_mode}, but model is not given, despite being required")
        print(f"Running forward pass...")
        loss = criterion_dbmtl(list(contributing_losses.values()), model, optimizer, preference_weights, dbmtl_epsilon)
        # Backward pass and optimization are performed inside DB-MTL loss object
        
    else:
        criterion_sum = SummedLoss(device)
        loss = criterion_sum(list(contributing_losses.values()))
        if sum_mode not in ["arithmetic", "sum", "simple", "linear"]: 
            print(f"Unrecognized sum_mode ({sum_mode}); defaulting to simple summation.")
        # Optimisation
        loss.backward()
        optimizer.step()

    # Track individual losses
    keys = ["ne", "os", "cc", "ov", "mu", "pn", "ne_ov", "os_ov", "cc_pn"]
    detached_losses = {}
    for key in keys:
        if contributing_losses.get(key) is not None:
            step_term_loss = contributing_losses[key].detach().cpu().numpy()
        elif spectator_losses.get(key) is not None:
            step_term_loss = spectator_losses[key].detach().cpu().numpy()
        elif unnecessary_losses.get(key) is not None:
            step_term_loss = unnecessary_losses[key].detach().cpu().numpy()
        elif blank_losses.get(key) is not None:
            step_term_loss = blank_losses[key].detach().cpu().numpy()
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
    # Filter the losses based on whether they contribute to the summed loss
    filtered_losses = filter_losses(optimizer, loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, 
                                    loss_ne_ov, loss_os_ov, loss_cc_pn, non_contributing_losses, squeeze=False)
    contributing_losses, unnecessary_losses, blank_losses, spectator_losses = filtered_losses
    
    # Backward pass
    grads = []
    for key, loss in contributing_losses.items():
        optimizer.zero_grad()  # Clear previous gradients
        try:
            loss.backward(retain_graph=True)  # Retain graph for backpropagation
        except Exception as e:
            raise Exception(f"Contributing loss {key} of type {type(loss)} produced the following exception during backpropagation: \n\t{e}")
        grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in model.parameters()])
        grads.append(grad)

    grads = torch.stack(grads, dim=0)  # Stack gradients

    # Perform backward pass on spectator losses
    for loss in spectator_losses.values():
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
    total_loss = sum(list(contributing_losses.values()))

    # Track the loss values for graphing purposes
    keys = ["ne", "os", "cc", "ov", "mu", "pn", "ne_ov", "os_ov", "cc_pn"]
    scalarized_losses = {}
    for key in keys:
        if contributing_losses.get(key) is not None:
            scalar_loss = to_scalar(contributing_losses[key])
        elif spectator_losses.get(key) is not None:
            scalar_loss = to_scalar(spectator_losses[key])
        elif unnecessary_losses.get(key) is not None:
            scalar_loss = to_scalar(unnecessary_losses[key])
        elif blank_losses.get(key) is not None:
            scalar_loss = to_scalar(blank_losses[key])
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
