import torch
from .solvers.procrustes_solver import ProcrustesSolver
from ..config import load_config, Config

def to_scalar(value):
    # Helper function that converts one-item Torch tensors into Python scalars (e.g. float)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.item()
        else:
            print("Cannot apply .item() to a tensor with more than one element.")
    return value

def track_loss(tracked_losses, key, loss_val):
    if key not in tracked_losses.keys():
        tracked_losses[key] = []
    tracked_losses[key].append(loss_val)

def track_losses(tracked_losses, loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_mu = None, 
                 loss_pn = None, loss_ne_ov = None, loss_os_ov = None, loss_cc_pn = None, loss_total = None): 
    # Track losses
    if loss_ne is not None:
        track_loss(tracked_losses, "Nuclei Encapsulation Loss", loss_ne)
    if loss_os is not None:
        track_loss(tracked_losses, "Oversegmentation Loss", loss_os)
    if loss_cc is not None:
        track_loss(tracked_losses, "Cell Calling Loss", loss_cc)
    if loss_ov is not None:
        track_loss(tracked_losses, "Overlap Loss", loss_ov)
    if loss_mu is not None:
        track_loss(tracked_losses, "Multiple Assignment Loss", loss_mu)
    if loss_pn is not None:
        track_loss(tracked_losses, "Pos-Neg Marker Loss", loss_pn)
    if loss_ne_ov is not None:
        track_loss(tracked_losses, "Combined Nuclei Encapsulation and Overlap Loss", loss_ne_ov)
    if loss_os_ov is not None:
        track_loss(tracked_losses, "Combined Oversegmentation and Overlap Loss", loss_os_ov)
    if loss_cc_pn is not None:
        track_loss(tracked_losses, "Combined Cell Calling and Marker Loss", loss_cc_pn)
    if loss_total is not None:
        track_loss(tracked_losses, "Total Loss", loss_total)

def filter_non_contributing(loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_mu = None, loss_pn = None, 
                            loss_ne_ov = None, loss_os_ov = None, loss_cc_pn = None, non_contributing_losses = (), assign_none=False): 
    # Remove non-contributing losses
    terms = [loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, loss_ne_ov, loss_os_ov, loss_cc_pn]
    keys = ["ne", "os", "cc", "ov", "mu", "pn", "ne_ov", "os_ov", "cc_pn"]
    contributing_terms = {}
    blank_terms = {}
    spectator_terms = {}

    for key, term in zip(keys, terms):
        if key in non_contributing_losses and term is not None:
            spectator_terms[key] = term
        elif key not in non_contributing_losses and term is not None:
            contributing_terms[key] = term
        else:
            blank_terms[key] = term

    return (contributing_terms, blank_terms, spectator_terms)

def filter_unnecessary(contributing_terms):
    # Removes loss terms that are already covered by a combined loss term
    
    keys = list(contributing_terms.keys())
    unnecessary_keys = []
    necessary_terms = {}
    unnecessary_terms = {}

    for key in keys:
        if "_" in key:
            unnecessary_keys.extend(key.split("_"))
    unnecessary_keys = list(tuple(unnecessary_keys))

    for key, term in contributing_terms.items():
        if key in unnecessary_keys:
            unnecessary_terms[key] = term
        else:
            necessary_terms[key] = term

    return (necessary_terms, unnecessary_terms)

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
    for loss in contributing_terms.values():
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward(retain_graph=True)  # Retain graph for backpropagation
        grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in model.parameters()])
        grads.append(grad)

    grads = torch.stack(grads, dim=0)  # Stack gradients

    # Perform backward pass on spectator losses
    for loss in spectator_terms.values():
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

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
