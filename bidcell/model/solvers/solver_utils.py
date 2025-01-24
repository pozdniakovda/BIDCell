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
                            assign_none = False, preference_weights = {}): 
    # Remove non-contributing losses
    terms = [loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, loss_ne_ov, loss_os_ov, loss_cc_pn]
    keys = ["ne", "os", "cc", "ov", "mu", "pn", "ne_ov", "os_ov", "cc_pn"]
    
    contributing_terms = {}
    contributing_weights = {}
    
    blank_terms = {}
    blank_weights = {}

    spectator_terms = {}
    spectator_weights = {}

    for key, term in zip(keys, terms):
        weight = preference_weights.get(key)
        if key in non_contributing_losses and term is not None:
            spectator_terms[key] = term
            spectator_weights[key] = weight
        elif key not in non_contributing_losses and term is not None:
            contributing_terms[key] = term
            contributing_weights[key] = weight
        else:
            blank_terms[key] = term
            blank_weights[key] = weight

    output = (contributing_terms, contributing_weights, 
              blank_terms, blank_weights, 
              spectator_terms, spectator_weights)

    return output

def filter_unnecessary(contributing_terms, preference_weights = {}):
    # Removes loss terms that are already covered by a combined loss term
    
    keys = list(contributing_terms.keys())
    unnecessary_keys = []
    
    necessary_terms = {}
    necessary_weights = {}
    
    unnecessary_terms = {}
    unnecessary_weights = {}

    for key in keys:
        if "_" in key:
            unnecessary_keys.extend(key.split("_"))
    unnecessary_keys = list(tuple(unnecessary_keys))

    for key, term in contributing_terms.items():
        weight = preference_weights.get(key)
        if key in unnecessary_keys:
            unnecessary_terms[key] = term
            unnecessary_weights[key] = weight
        else:
            necessary_terms[key] = term
            necessary_weights[key] = weight

    return (necessary_terms, necessary_weights, unnecessary_terms, unnecessary_weights)
