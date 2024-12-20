import argparse
import logging
import warnings
import math
import sys
import os
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from .solvers.procrustes_solver import ProcrustesSolver

from .dataio.dataset_input import DataProcessing
from .model.losses import (
    CellCallingLoss,
    NucleiEncapsulationLoss,
    OverlapLoss,
    MultipleAssignmentLoss,
    OversegmentationLoss,
    PosNegMarkerLoss
)
from .model.combined_losses import (
    NucEncapOverlapLoss,
    OversegOverlapLoss,
    CellCallingMarkerLoss
)
from .model.model import SegmentationModel as Network
from .utils.utils import (
    get_experiment_id,
    make_dir,
    save_fig_outputs,
)
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

def sum_losses(loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_mu = None, loss_pn = None, 
               loss_ne_ov = None, loss_os_ov = None, loss_cc_pn = None, non_contributing_losses = ()): 
    # Remove non-contributing losses
    if len(non_contributing_losses) > 0:
        args = filter_non_contributing(loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, 
                                       loss_ne_ov, loss_os_ov, loss_cc_pn, 
                                       non_contributing_losses, assign_none=False)
        contributing_terms, blank_terms, spectator_terms = args
        
        loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, loss_ne_ov, loss_os_ov, loss_cc_pn = args

    # Dynamically calculate total loss
    if loss_ne_ov is not None:
        if loss_cc_pn is not None:
            loss = loss_ne_ov + loss_os + loss_cc_pn + loss_mu
        else: 
            loss = loss_ne_ov + loss_os + loss_cc + loss_pn + loss_mu
    elif loss_os_ov is not None: 
        if loss_cc_pn is not None:
            loss = loss_ne + loss_os_ov + loss_cc_pn + loss_mu
        else: 
            loss = loss_ne + loss_os_ov + loss_cc + loss_pn + loss_mu
    elif loss_cc_pn is not None:
        loss = loss_ne + loss_os + loss_ov + loss_cc_pn + loss_mu
    else: 
        loss = loss_ne + loss_os + loss_ov + loss_cc + loss_pn + loss_mu

    return loss

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

def plot_overlaid_losses(total_loss_vals, total_loss_ma, other_loss_vals, other_loss_ma, total_epochs, 
                         train_loader_len, experiment_path, solver_title, log_scale=True, 
                         rescaling=True, show_moving_averages=True):
    # Plots all the losses on one graph

    plt.figure(figsize=(18, 8))

    if rescaling:
        divisor = max(total_loss_vals) / 1000 if max(total_loss_vals) != 0 else 1
        total_loss_vals = np.divide(total_loss_vals, divisor)
    plt.plot(total_loss_vals, label="Total Loss", linewidth=1)

    divisors = {}
    for label, loss_vals in other_loss_vals.items():
        if loss_vals is not None:
            if len(loss_vals) > 0:
                divisor = max(loss_vals) / 1000 if max(loss_vals) != 0 else 1
                if rescaling:
                    divisors[label] = divisor
                    loss_vals = np.divide(loss_vals, divisor)
                plt.plot(loss_vals, label=label, linewidth=0.5, alpha=0.5)

    if show_moving_averages:
        ma_loss_vals, ma_window_width = total_loss_ma
        if rescaling:
            divisor = max(ma_loss_vals) / 1000 if max(ma_loss_vals) != 0 else 1
            ma_loss_vals = np.divide(ma_loss_vals, divisor)
        plt.plot(ma_loss_vals, label=f"Total Loss (moving average, {ma_window_width})", linewidth=2)

        for label, loss_ma_tuple in other_loss_ma.items():
            loss_ma, ma_window_width = loss_ma_tuple
            if loss_ma is not None:
                if len(loss_ma) > 0:
                    if rescaling:
                        divisor = divisors[label]
                        loss_ma = np.divide(loss_ma, divisor)
                    plt.plot(loss_ma, label=f"{label} (moving average, {ma_window_width})", linewidth=1, alpha=0.5)

    for epoch in range(total_epochs):
        plt.axvline(x=epoch * train_loader_len, color="r", linestyle="--", alpha=0.5)

    if log_scale:
        plt.yscale("log")

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    title = f"Training Loss with {solver_title}"
    if rescaling:
        title = title + " (rescaled to max=1000)"
    plt.legend()
    plt.tight_layout()

    filename = "training_losses_overlaid.pdf" if not rescaling else "training_losses_overlaid_rescaled.pdf"
    save_path = os.path.join(experiment_path, filename)
    plt.savefig(save_path)
    #plt.show()

def plot_loss(loss_vals, ma_loss_vals, label, total_epochs, experiment_path, train_loader_len,
              solver_title, log_scale=True, rescaling=True, show_moving_averages=True):
    # Plots a single objective's values over the course of the training cycle
    if loss_vals is not None:
        if len(loss_vals) > 0:
            ma_loss_vals, ma_window_width = ma_loss_vals
            if rescaling:
                divisor = max(loss_vals) / 1000 if max(loss_vals) != 0 else 1
                loss_vals = np.divide(loss_vals, divisor)
                if show_moving_averages:
                    ma_loss_vals = np.divide(ma_loss_vals, divisor)
            
            plt.figure(figsize=(18, 8))
            plt.plot(loss_vals, label=label, linewidth=0.5)
            if show_moving_averages:
                plt.plot(ma_loss_vals, label=f"{label} (moving average, {ma_window_width})", linewidth=2)
            
            for epoch in range(total_epochs):
                plt.axvline(x=epoch * train_loader_len, color="r", linestyle="--")
            
            if log_scale:
                plt.yscale("log")
            
            plt.xlabel("Training Step")
            plt.ylabel("Loss")

            title = f"{label} During Training with {solver_title}"
        
            if rescaling:
                title = title + " (rescaled to max=1000)"
            plt.title(title)
            plt.tight_layout()
            
            underscored_label = "_".join(label.lower().split(" "))
            filename = f"training_{underscored_label}.pdf" if not rescaling else f"training_{underscored_label}_rescaled.pdf"
            plt.savefig(os.path.join(experiment_path, filename))
            #plt.show()

def get_ma_losses(losses, window_width=None):
    # Calculate loss moving averages as 2.5% increments (e.g. 10 epochs x 1000 steps/epoch = 10,000 steps, i.e. 250 steps per point. 
    ma_losses = {}
    
    for loss_name, loss_vals in losses.items():
        if len(loss_vals) == 0: 
            ma_losses[loss_name] = ([], 0)
            continue

        if window_width is None:
            window_width = max(1, int(len(loss_vals) / 40))  # Ensure window width is at least 1
        loss_vals = np.array(loss_vals)
    
        try: 
            moving_averages = np.convolve(loss_vals, np.ones(window_width) / window_width, mode="valid") # main convolution
            moving_averages = np.concatenate([np.full(window_width - 1, moving_averages[0]), moving_averages]) # padding
        except Exception as e:
            raise Exception(f"Error during moving average calculation for {loss_name}: {e}")
    
        ma_losses[loss_name] = (moving_averages, window_width)

    return ma_losses

def plot_losses(losses, ma_losses, combine_ne_ov, combine_os_ov, combine_cc_pn, total_epochs, 
                train_loader_len, experiment_path, solver_title, log_scale=True):
    # Plot losses
    print(f"Graphing overlaid losses...")

    # Plot all losses on one graph
    total_loss_vals = losses["Total Loss"]
    total_loss_ma = ma_losses["Total Loss"]
    
    keys = ["Multiple Assignment Loss"]
    if combine_ne_ov:
        keys.extend(["Combined Nuclei Encapsulation and Overlap Loss", "Oversegmentation Loss"])
    elif combine_os_ov:
        keys.extend(["Combined Oversegmentation and Overlap Loss", "Nuclei Encapsulation Loss"])
    else:
        keys.extend(["Nuclei Encapsulation Loss", "Oversegmentation Loss", "Overlap Loss"])
    
    if combine_cc_pn:
        keys.append("Combined Cell Calling and Marker Loss")
    else:
        keys.extend(["Cell Calling Loss", "Pos-Neg Marker Loss"])
    
    other_loss_vals = {key:losses[key] for key in keys}
    other_loss_ma = {key:ma_losses[key] for key in keys}

    plot_overlaid_losses(total_loss_vals, total_loss_ma, other_loss_vals, other_loss_ma, total_epochs, 
                         train_loader_len, experiment_path, solver_title, log_scale, rescaling=False)

    # Plot individual losses
    print(f"Graphing total loss...")
    plot_loss(losses["Total Loss"], ma_losses["Total Loss"], "Total Loss", total_epochs, experiment_path, 
              train_loader_len, solver_title, log_scale, rescaling=False)
    print(f"Graphing individual losses...")
    for key in keys:
        plot_loss(losses[key], ma_losses[key], key, total_epochs, experiment_path, 
                  train_loader_len, solver_title, log_scale, rescaling=False)

    # Repeat for rescaled versions
    print(f"Graphing overlaid rescaled losses...")
    plot_overlaid_losses(total_loss_vals, total_loss_ma, other_loss_vals, other_loss_ma, total_epochs, 
                         train_loader_len, experiment_path, solver_title, log_scale, rescaling=True)
    print(f"Graphing rescaled total loss...")
    plot_loss(losses["Total Loss"], ma_losses["Total Loss"], "Total Loss", total_epochs, experiment_path, 
              train_loader_len, solver_title, log_scale, rescaling=True)
    print(f"Graphing rescaled individual losses...")
    for key in keys:
        plot_loss(losses[key], ma_losses[key], key, total_epochs, experiment_path, 
                  train_loader_len, solver_title, log_scale, rescaling=True)

def get_weighting_ratio(loss1, loss2, criterion_loss1, criterion_loss2, weights1, weights2, 
                        weights1_names, weights2_names, input_shape, combine_mode, logging):
    loss1 = to_scalar(loss1)
    loss2 = to_scalar(loss2)
    ratio = None
    
    if combine_mode == "top":
        logging.info(f"Dynamically adjusting weights based on loss values at first step.")
        if loss1 != 0 and loss2 != 0:
            ratio = loss1 / loss2
            weights2 = [weight2 * ratio for weight2 in weights2]
            message = f"loss1={loss1}, loss2={loss2}, ratio={ratio}"
            for weight_name, weight_val in zip(weights2_names, weights2):
                message = f"{message}; {weight_name} adjusted to new value of {weight_val}"
            logging.info(message)
        
    elif combine_mode == "max": 
        logging.info(f"Dynamically adjusting weights based on maximum theoretical loss values.")
        max_loss1 = criterion_loss1.get_max(input_shape, *weights1)
        max_loss2 = criterion_loss2.get_max(input_shape, *weights2)
        if max_loss1 != 0 and max_loss2 != 0:
            ratio = max_loss1 / max_loss2
            weights2 = [weight2 * ratio for weight2 in weights2]
            message = f"max_loss1={max_loss1}, max_loss2={max_loss2}, ratio={ratio}"
            for weight_name, weight_val in zip(weights2_names, weights2):
                message = f"{message}; {weight_name} adjusted to new value of {weight_val}"
            logging.info(message)
        
    else:
        raise ValueError(f"combine_mode must be top, max, or static, but was given as {combine_mode}")

    return ratio, weights1, weights2

def get_solver_title(selected_solver = None, starting_solver = None, ending_solver = None, epochs_before_switch = 0, dynamic_solvers = False):
    # Generates a title fragment referencing the solver(s) that were used during training
    
    if dynamic_solvers:        
        if "procrustes" in starting_solver.lower():
            starting_scale_mode = "median" if "median" in starting_solver else "rmse" if "rmse" in starting_solver else "min"
            starting_solver_title = f"Procrustes Method (scaling mode: {starting_scale_mode})"
        else:
            starting_solver_title = f"Default Method"

        if "procrustes" in ending_solver.lower():
            ending_scale_mode = "median" if "median" in ending_solver else "rmse" if "rmse" in ending_solver else "min"
            ending_solver_title = f"Procrustes Method (scaling mode: {ending_scale_mode})"
        else:
            ending_solver_title = f"Default Method"

        solver_title = f"{starting_solver_title} (epochs 1-{epochs_before_switch}) to {ending_solver_title} (epochs {epochs_before_switch+1} onwards)"
        
    elif "procrustes" in selected_solver:
        scale_mode = "median" if "median" in selected_solver else "rmse" if "rmse" in selected_solver else "min"
        solver_title = f"Procrustes Method (scaling mode: {scale_mode})"
        
    else:
        solver_title = "Default Method"

    return solver_title

def train(config: Config, learning_rate = None, selected_solver = None):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create experiment directories
    resume_epoch = None  # could be added
    resume_step = 0
    if resume_epoch is None:
        make_new = True
    else:
        make_new = False

    if config.training_params.model_freq <= config.testing_params.test_step:
        model_freq = config.training_params.model_freq
    else:
        model_freq = config.testing_params.test_step

    # Set up the model
    logging.info("Initialising model")

    atlas_exprs = pd.read_csv(config.files.fp_ref, index_col=0)
    n_genes = atlas_exprs.shape[1] - 3
    print("Number of genes: %d" % n_genes)

    if config.model_params.name != "custom":
        model = smp.Unet(
            encoder_name=config.model_params.name,
            encoder_weights=None,
            in_channels=n_genes,
            classes=2,
        )
    else:
        model = Network(n_channels=n_genes)

    model = model.to(device)

    # Dataloader
    logging.info("Preparing data")

    train_dataset = DataProcessing(
        config,
        isTraining=True,
        total_steps=config.training_params.total_steps,
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True
    )

    n_train_examples = len(train_loader)
    logging.info("Total number of training examples: %d" % n_train_examples)

    # Loss weights
    ne_weight = config.training_params.ne_weight
    os_weight = config.training_params.os_weight
    cc_weight = config.training_params.cc_weight
    ov_weight = config.training_params.ov_weight
    mu_weight = config.training_params.mu_weight
    pos_weight = config.training_params.pos_weight
    neg_weight = config.training_params.neg_weight

    # Overlap loss preferences
    ov_distance_scaling = config.training_params.ov_distance_scaling
    ov_intensity_weighting = config.training_params.ov_intensity_weighting
    
    # Loss functions
    criterion_ne = NucleiEncapsulationLoss(ne_weight, device)
    criterion_os = OversegmentationLoss(os_weight, device)
    criterion_cc = CellCallingLoss(cc_weight, device)
    criterion_ov = OverlapLoss(ov_weight, device)
    criterion_mu = MultipleAssignmentLoss(mu_weight, device)
    criterion_pn = PosNegMarkerLoss(pos_weight, neg_weight, device)

    # Combined loss functions if desired
    combine_ne_ov = config.training_params.combine_ne_ov
    combine_os_ov = config.training_params.combine_os_ov
    combine_cc_pn = config.training_params.combine_cc_pn
    if combine_ne_ov and combine_os_ov: 
        raise Exception(f"combine_ne_ov and combine_os_ov were both set to True, but they are "
                        "mutually exclusive because they both use OverlapLoss.")

    combine_ne_ov_mode = config.training_params.combine_ne_ov_mode
    combine_os_ov_mode = config.training_params.combine_os_ov_mode
    combine_cc_pn_mode = config.training_params.combine_cc_pn_mode

    criterion_ne_ov = NucEncapOverlapLoss(ne_weight, ov_weight, device) if combine_ne_ov else None
    criterion_os_ov = OversegOverlapLoss(os_weight, ov_weight, device) if combine_os_ov else None
    criterion_cc_pn = CellCallingMarkerLoss(cc_weight, pos_weight, neg_weight, device) if combine_cc_pn else None

    # Non-contributing losses
    non_contributing_losses = config.training_params.non_contributing_losses

    # Solver and learning rate
    if selected_solver is None: 
        selected_solver = config.training_params.solver
        starting_solver = config.training_params.starting_solver
        ending_solver = config.training_params.ending_solver
        epochs_before_switch = config.training_params.epochs_before_switch
    dynamic_solvers = starting_solver != "" and ending_solver != "" and epochs_before_switch > 0
    if learning_rate is None:
        learning_rate = config.training_params.learning_rate

    # Generate path for saving outputs
    timestamp = get_experiment_id(
        make_new,
        config.experiment_dirs.dir_id,
        config.files.data_dir,
    )
    if not dynamic_solvers: 
        experiment_path = os.path.join(config.files.data_dir, "model_outputs", f"{timestamp}_{selected_solver}_lr-{learning_rate}")
    else:
        experiment_path = os.path.join(config.files.data_dir, "model_outputs", f"{timestamp}_{starting_solver}-to-{ending_solver}_switched-after-{epochs_before_switch}-epochs_lr-{learning_rate}")
    make_dir(experiment_path + "/" + config.experiment_dirs.model_dir)
    make_dir(experiment_path + "/" + config.experiment_dirs.samples_dir)

    # Optimiser
    print(f"Current learning rate: {learning_rate}")
    if config.training_params.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-8,
        )
    elif config.training_params.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(config.training_params.beta1, config.training_params.beta2),
            weight_decay=config.training_params.weight_decay,
        )
    else:
        sys.exit("Select optimiser from rmsprop or adam")

    global_step = 0
    losses = {}

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = (
        lambda x: (
            ((1 + math.cos(x * math.pi / config.training_params.total_epochs)) / 2)
            ** 1.0
        )
        * 0.95
        + 0.05
    )  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = global_step

    # Starting epoch
    if resume_epoch is not None:
        initial_epoch = resume_epoch
    else:
        initial_epoch = 0

    # Restore saved model
    if resume_epoch is not None:
        load_path = (
            experiment_path
            + "/"
            + config.experiment_dirs.model_dir
            + "/epoch_%d_step_%d.pth" % (resume_epoch, resume_step)
        )
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        assert epoch == resume_epoch
        print("Resume training, successfully loaded " + load_path)

    if dynamic_solvers:
        logging.info(f"Begin training using {starting_solver} for {epochs_before_switch} epochs, followed by {ending_solver} thereafter")
        
        if "procrustes" in starting_solver.lower():
            starting_scale_mode = "median" if "median" in starting_solver else "rmse" if "rmse" in starting_solver else "min"
            starting_solver_title = f"Procrustes Method (scaling mode: {starting_scale_mode})"
        else:
            starting_solver_title = f"Default Method"

        if "procrustes" in ending_solver.lower():
            ending_scale_mode = "median" if "median" in ending_solver else "rmse" if "rmse" in ending_solver else "min"
            ending_solver_title = f"Procrustes Method (scaling mode: {ending_scale_mode})"
        else:
            ending_solver_title = f"Default Method"

        solver_title = f"{starting_solver_title} (epochs 1-{epochs_before_switch}) to {ending_solver_title} (epochs {epochs_before_switch+1} onwards)"
        
    elif "procrustes" in selected_solver:
        logging.info("Begin training using Procrustes method")
        scale_mode = "median" if "median" in selected_solver else "rmse" if "rmse" in selected_solver else "min"
        solver_title = f"Procrustes Method (scaling mode: {scale_mode})"
        
    else:
        solver_title = "Default Method"
        logging.info("Begin training using default method")

    model = model.train()

    lrs = []
    scale_mode = ""
    
    is_first_step = True
    for epoch in range(initial_epoch, config.training_params.total_epochs):
        # Define current solver
        if dynamic_solvers:
            current_solver = starting_solver if epoch < epochs_before_switch else ending_solver
        else:
            current_solver = selected_solver

        cur_lr = optimizer.param_groups[0]["lr"]
        print("\nEpoch =", (epoch + 1), " lr =", cur_lr, " solver =", current_solver)

        for step_epoch, (
            batch_x313,
            batch_n,
            batch_sa,
            batch_pos,
            batch_neg,
            coords_h1,
            coords_w1,
            nucl_aug,
            expr_aug_sum,
        ) in enumerate(train_loader):
            # Permute channels axis to batch axis
            batch_x313 = batch_x313[0, :, :, :, :].permute(3, 2, 0, 1)
            batch_sa = batch_sa.permute(3, 0, 1, 2)
            batch_pos = batch_pos.permute(3, 0, 1, 2)
            batch_neg = batch_neg.permute(3, 0, 1, 2)
            batch_n = batch_n.permute(3, 0, 1, 2)

            if batch_x313.shape[0] == 0:
                # Save the model periodically
                if (step_epoch % model_freq) == 0:
                    save_path = (
                        experiment_path
                        + "/"
                        + config.experiment_dirs.model_dir
                        + "/epoch_%d_step_%d.pth" % (epoch + 1, step_epoch)
                    )
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        save_path,
                    )
                    logging.info("Model saved: %s" % save_path)
                continue

            # Transfer to GPU
            batch_x313 = batch_x313.to(device)
            batch_sa = batch_sa.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)
            batch_n = batch_n.to(device)
            expr_aug_sum = expr_aug_sum.to(device)

            optimizer.zero_grad()

            seg_pred = model(batch_x313)

            # Compute individual losses as appropriate
            loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn = None, None, None, None, None, None
            ov_combined = combine_ne_ov or combine_os_ov
            if is_first_step or not combine_ne_ov:
                loss_ne = criterion_ne(seg_pred, batch_n, ne_weight)
            if is_first_step or not combine_os_ov:
                loss_os = criterion_os(seg_pred, batch_n, os_weight)
            if is_first_step or not ov_combined:
                loss_ov = criterion_ov(seg_pred, batch_n, ov_weight)
            if is_first_step or not combine_cc_pn:
                loss_cc = criterion_cc(seg_pred, batch_sa, cc_weight)
                loss_pn = criterion_pn(seg_pred, batch_pos, batch_neg, pos_weight, neg_weight)
            loss_mu = criterion_mu(expr_aug_sum, batch_sa, mu_weight)
            
            if is_first_step:
                if combine_ne_ov or combine_os_ov or combine_cc_pn:
                    logging.info(f"Computing all constituent losses for first step; subsequent steps will use combined losses.")

            # If selected, adjust loss weights to compensate for different magnitudes of initial values
            if combine_ne_ov:
                if combine_ne_ov_mode != "static" and is_first_step:
                    ratio, weights1, weights2 = get_weighting_ratio(loss_ne, loss_ov, criterion_ne, criterion_ov, [ne_weight], [ov_weight], 
                                                                    ["ne_weight"], ["ov_weight"], seg_pred.shape, combine_ne_ov_mode, logging)
                    ne_weight, ov_weight = weights1[0], weights2[0]
                    loss_ne, loss_ov = None, None # reset to make sure they are not used in grads
            if combine_os_ov:
                if combine_os_ov_mode != "static" and is_first_step:
                    ratio, weights1, weights2 = get_weighting_ratio(loss_os, loss_ov, criterion_os, criterion_ov, [os_weight], [ov_weight], 
                                                                    ["os_weight"], ["ov_weight"], seg_pred.shape, combine_os_ov_mode, logging)
                    os_weight, ov_weight = weights1[0], weights2[0]
                    loss_os, loss_ov = None, None # reset to make sure they are not used in grads
            if combine_cc_pn:
                if combine_cc_pn_mode != "static" and is_first_step:
                    ratio, weights1, weights2 = get_weighting_ratio(loss_cc, loss_pn, criterion_cc, criterion_pn, [cc_weight], [pos_weight, neg_weight], 
                                                                    ["cc_weight"], ["pos_weight", "neg_weight"], seg_pred.shape, combine_cc_pn_mode, logging)
                    cc_weight = weights1[0]
                    pos_weight, neg_weight = weights2
                    loss_cc, loss_pn = None, None # reset to make sure they are not used in grads
            
            is_first_step = False # first step special case handling is concluded
            
            # Calculate combined losses if required
            loss_ne_ov = criterion_ne_ov(seg_pred, batch_n, ne_weight, ov_weight) if combine_ne_ov else None
            loss_os_ov = criterion_os_ov(seg_pred, batch_n, os_weight, ov_weight) if combine_os_ov else None
            loss_cc_pn = criterion_cc_pn(seg_pred, batch_sa, batch_pos, batch_neg, cc_weight, pos_weight, neg_weight) if combine_cc_pn else None

            # Apply the Procrustes method
            if "procrustes" in current_solver:
                scale_mode = "median" if "median" in current_solver else "rmse" if "rmse" in current_solver else "min"
                total_loss = procrustes_method(model = model, 
                                               optimizer = optimizer, 
                                               tracked_losses = losses, 
                                               loss_ne = loss_ne, 
                                               loss_os = loss_os, 
                                               loss_cc = loss_cc, 
                                               loss_ov = loss_ov, 
                                               loss_mu = loss_mu, 
                                               loss_pn = loss_pn, 
                                               loss_ne_ov = loss_ne_ov, 
                                               loss_os_ov = loss_os_ov, 
                                               loss_cc_pn = loss_cc_pn, 
                                               scale_mode = "min", 
                                               non_contributing_losses = non_contributing_losses)
            else: 
                total_loss = default_solver(optimizer = optimizer, 
                                            tracked_losses = losses, 
                                            loss_ne = loss_ne, 
                                            loss_os = loss_os, 
                                            loss_cc = loss_cc, 
                                            loss_ov = loss_ov, 
                                            loss_mu = loss_mu,
                                            loss_pn = loss_pn, 
                                            loss_ne_ov = loss_ne_ov, 
                                            loss_os_ov = loss_os_ov, 
                                            loss_cc_pn = loss_cc_pn, 
                                            non_contributing_losses = non_contributing_losses)
            
            if (global_step % config.training_params.sample_freq) == 0:
                coords_h1 = coords_h1.detach().cpu().squeeze().numpy()
                coords_w1 = coords_w1.detach().cpu().squeeze().numpy()
                sample_seg = seg_pred.detach().cpu().numpy()
                sample_n = nucl_aug.detach().cpu().numpy()
                sample_sa = batch_sa.detach().cpu().numpy()
                sample_expr = expr_aug_sum.detach().cpu().numpy()
                patch_fp = (
                    experiment_path
                    + "/"
                    + config.experiment_dirs.samples_dir
                    + "/epoch_%d_%d_%d_%d.png"
                    % (epoch + 1, step_epoch, coords_h1, coords_w1)
                )

                save_fig_outputs(sample_seg, sample_n, sample_sa, sample_expr, patch_fp)

                print(
                    "Epoch[{}/{}], Step[{}], Total Loss:{:.4f}".format(
                        epoch + 1,
                        config.training_params.total_epochs,
                        step_epoch,
                        total_loss,
                    )
                )

            # Save model
            if (step_epoch % model_freq) == 0:
                save_path = (
                    experiment_path
                    + "/"
                    + config.experiment_dirs.model_dir
                    + "/epoch_%d_step_%d.pth" % (epoch + 1, step_epoch)
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    save_path,
                )
                logging.info("Model saved: %s" % save_path)

            global_step += 1

        # Update and append current LR
        scheduler.step()
        lrs.append(cur_lr)

    # Graph the losses
    total_epochs = config.training_params.total_epochs
    train_loader_len = len(train_loader)
    log_scale = config.training_params.log_scale
    ma_losses = get_ma_losses(losses)
    solver_title = get_solver_title(selected_solver, starting_solver, ending_solver, 
                                    epochs_before_switch, dynamic_solvers)
    plot_losses(losses, ma_losses, combine_ne_ov, combine_os_ov, combine_cc_pn, total_epochs, 
                train_loader_len, experiment_path, solver_title, log_scale)

    logging.info("Training finished")

    return losses, ma_losses, experiment_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")

    args = parser.parse_args()
    config = load_config(args.config_dir)

    train(config)
