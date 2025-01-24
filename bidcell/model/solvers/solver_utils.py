import numpy as np
import torch

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
                            loss_ne_ov = None, loss_os_ov = None, loss_cc_pn = None, non_contributing_losses = (), 
                            assign_none = False): 
    # Remove non-contributing losses
    terms = [loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, loss_ne_ov, loss_os_ov, loss_cc_pn]
    keys = ["ne", "os", "cc", "ov", "mu", "pn", "ne_ov", "os_ov", "cc_pn"]
    
    contributing_losses = {}
    blank_losses = {}
    spectator_losses = {}

    for key, term in zip(keys, terms):
        if key in non_contributing_losses and term is not None:
            spectator_losses[key] = term
        elif key not in non_contributing_losses and term is not None:
            contributing_losses[key] = term
        else:
            blank_losses[key] = term

    return contributing_losses, blank_losses, spectator_losses

def filter_unnecessary(contributing_losses):
    # Removes loss terms that are already covered by a combined loss term
    
    keys = list(contributing_losses.keys())
    unnecessary_keys = []
    
    necessary_losses = {}    
    unnecessary_losses = {}

    for key in keys:
        if "_" in key:
            unnecessary_keys.extend(key.split("_"))
    unnecessary_keys = list(tuple(unnecessary_keys))

    for key, term in contributing_losses.items():
        if key in unnecessary_keys:
            unnecessary_losses[key] = term
        else:
            necessary_losses[key] = term

    return necessary_losses, unnecessary_losses

def filter_losses(optimizer, loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_mu = None, 
                  loss_pn = None, loss_ne_ov = None, loss_os_ov = None, loss_cc_pn = None, 
                  non_contributing_losses=(), squeeze = True):
    # Separates losses based on whether they should contribute to the summed loss
    
    if squeeze:
        loss_ne = loss_ne.squeeze() if loss_ne is not None else None
        loss_os = loss_os.squeeze() if loss_os is not None else None
        loss_cc = loss_cc.squeeze() if loss_cc is not None else None
        loss_ov = loss_ov.squeeze() if loss_ov is not None else None
        loss_mu = loss_mu.squeeze() if loss_mu is not None else None
        loss_pn = loss_pn.squeeze() if loss_pn is not None else None
        
        loss_ne_ov = loss_ne_ov.squeeze() if loss_ne_ov is not None else None
        loss_os_ov = loss_os_ov.squeeze() if loss_os_ov is not None else None
        loss_cc_pn = loss_cc_pn.squeeze() if loss_cc_pn is not None else None

    # Filter based on whether a loss is designated as contributing
    contribution_args = filter_non_contributing(loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, 
                                                loss_ne_ov, loss_os_ov, loss_cc_pn, 
                                                non_contributing_losses, assign_none=False)
    contributing_losses, blank_losses, spectator_losses = contribution_args
    contributing_losses, unnecessary_losses = filter_unnecessary(contributing_losses)

    return contributing_losses, unnecessary_losses, blank_losses, spectator_losses
