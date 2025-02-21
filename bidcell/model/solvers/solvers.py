import torch
from .solver_utils import (
    to_scalar, 
    loss_key_conversion, 
    assign_loss, 
    assign_losses, 
    filter_losses,
)
from .procrustes_solver import ProcrustesSolver
from ..model.loss_summation import SummedLoss, STCHLoss, DBMTLLoss
from ...config import load_config, Config

def summed_solver(optimizer, device, tracked_losses, 
                  loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_mu = None, loss_pn = None, 
                  loss_ne_ov = None, loss_os_ov = None, loss_cc_pn = None, non_contributing_losses=(), 
                  preference_weights = None):
    # Default solver for summed losses

    # Filter the losses based on whether they contribute to the summed loss
    filtered_losses = filter_losses(optimizer, loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, 
                                    loss_ne_ov, loss_os_ov, loss_cc_pn, non_contributing_losses, squeeze=True)
    contributing_losses, unnecessary_losses, blank_losses, spectator_losses = filtered_losses

    # Sum the contributing losses
    criterion_sum = SummedLoss(device)
    loss = criterion_sum(list(contributing_losses.values()))
    if sum_mode not in ["arithmetic", "sum", "simple", "linear"]: 
        print(f"Unrecognized sum_mode ({sum_mode}); defaulting to simple summation.")

    # Optimisation
    loss.backward()
    optimizer.step()

    # Track individual losses
    step_total_loss = assign_losses(tracked_losses, contributing_losses, spectator_losses, 
                                    unnecessary_losses, blank_losses, loss)

    return step_total_loss

def stch_solver(optimizer, device, tracked_losses, 
                loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_mu = None, loss_pn = None, 
                loss_ne_ov = None, loss_os_ov = None, loss_cc_pn = None, non_contributing_losses=(), 
                preference_weights = None, ideal_vals = None, stch_mu = 1.0):
    # STCH Solver

    # Filter the losses based on whether they contribute to the summed loss
    filtered_losses = filter_losses(optimizer, loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, 
                                    loss_ne_ov, loss_os_ov, loss_cc_pn, non_contributing_losses, squeeze=True)
    contributing_losses, unnecessary_losses, blank_losses, spectator_losses = filtered_losses

    # Combine the losses using STCH
    criterion_stch = STCHLoss(preference_weights, device)
    loss = criterion_stch(list(contributing_losses.values()), preference_weights, ideal_vals, stch_mu)
    loss.backward()
    optimizer.step()

    # Track individual losses
    step_total_loss = assign_losses(tracked_losses, contributing_losses, spectator_losses, 
                                    unnecessary_losses, blank_losses, loss)

    return step_total_loss

def dbmtl_solver(optimizer, device, tracked_losses, model = None, 
                 loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_mu = None, loss_pn = None, 
                 loss_ne_ov = None, loss_os_ov = None, loss_cc_pn = None, non_contributing_losses=(), 
                 preference_weights = None, dbmtl_epsilon = None):
    # DB-MTL Solver

    # Filter the losses based on whether they contribute to the summed loss
    filtered_losses = filter_losses(optimizer, loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, 
                                    loss_ne_ov, loss_os_ov, loss_cc_pn, non_contributing_losses, squeeze=True)
    contributing_losses, unnecessary_losses, blank_losses, spectator_losses = filtered_losses

    # Apply DB-MTL method
    criterion_dbmtl = DBMTLLoss(preference_weights, device)
    if dbmtl_epsilon is None:
        raise Exception(f"sum_mode was set to {sum_mode}, but dbmtl_epsilon was not given (None)")
    if model is None:
        raise Exception(f"sum_mode was set to {sum_mode}, but model is not given, despite being required")
    log_losses, log_total_loss, total_loss = criterion_dbmtl(list(contributing_losses.values()), model, optimizer, preference_weights, dbmtl_epsilon)

    # Compute gradients for each task; use log-transformed losses
    grads = []
    optimizer.zero_grad()
    for loss in log_losses:
        if loss.item() != 0: 
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in model.parameters()])
            grads.append(grad)
    grads = torch.stack(grads, dim=0)  # Stack gradients

    # Normalize gradients to match the maximum gradient norm
    max_grad_norm = grads.norm(dim=1).max()
    normalized_grads = grads / grads.norm(dim=1, keepdim=True).clamp(min=dbmtl_epsilon) * max_grad_norm

    # Apply normalized gradients back to model parameters
    grad = normalized_grads.sum(dim=0)
    offset = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        _offset = offset + p.grad.numel()
        p.grad.data = grad[offset:_offset].view_as(p.grad)
        offset = _offset

    # Perform optimization step
    optimizer.step()

    # Track individual losses; untransformed total loss is used
    step_total_loss = assign_losses(tracked_losses, contributing_losses, spectator_losses, 
                                    unnecessary_losses, blank_losses, total_loss)

    return step_total_loss

def procrustes_method(model, optimizer, tracked_losses, loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_mu = None, 
                      loss_pn = None, loss_ne_ov = None, loss_os_ov = None, loss_cc_pn = None, scale_mode = "min", non_contributing_losses=()): 
    # Filter the losses based on whether they contribute to the summed loss
    filtered_losses = filter_losses(optimizer, loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, 
                                    loss_ne_ov, loss_os_ov, loss_cc_pn, non_contributing_losses, squeeze=False)
    contributing_losses, unnecessary_losses, blank_losses, spectator_losses = filtered_losses
    
    # Backward pass
    grads = []
    for key, loss in contributing_losses.items():
        if loss.item() != 0: 
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
        if loss.item() != 0: 
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
    step_total_loss = assign_losses(tracked_losses, contributing_losses, spectator_losses, 
                                    unnecessary_losses, blank_losses, total_loss)

    return step_total_loss
