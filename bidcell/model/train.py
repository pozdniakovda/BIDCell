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
    Oversegmentation,
    PosNegMarkerLoss,
    NucEncapOverlapLoss,
    CellCallingMarkerLoss
)
from .model.model import SegmentationModel as Network
from .utils.utils import (
    get_experiment_id,
    make_dir,
    save_fig_outputs,
)
from ..config import load_config, Config

def check_loss_args(loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_pn = None, loss_ne_ov = None, loss_cc_pn = None):
    # Check validity of given arguments
    if loss_ne_ov is None:
        if loss_ne is None and loss_ov is None:
            raise Exception(f"Missing loss_ne and loss_ov")
        elif loss_ne is None:
            raise Exception(f"Missing loss_ne.")
        elif loss_ov is None:
            raise Exception(f"Missing loss_ov.")
    elif loss_ne is not None and loss_ov is not None:
        warnings.warn(f"loss_ne and loss_ov were given, but they were ignored, as loss_ne_ov was also given.")
    elif loss_ne is not None:
        warnings.warn(f"loss_ne was given, but it was ignored, as loss_ne_ov was also given.")
    elif loss_ov is not None:
        warnings.warn(f"loss_ov was given, but it was ignored, as loss_ne_ov was also given.")

    if loss_cc_pn is None:
        if loss_cc is None and loss_pn is None:
            raise Exception(f"Missing loss_cc and loss_pn")
        elif loss_cc is None:
            raise Exception(f"Missing loss_cc.")
        elif loss_pn is None:
            raise Exception(f"Missing loss_pn.")
    elif loss_cc is not None and loss_pn is not None:
        warnings.warn(f"loss_cc and loss_pn were given, but they were ignored, as loss_cc_pn was also given.")
    elif loss_cc is not None:
        warnings.warn(f"loss_cc was given, but it was ignored, as loss_cc_pn was also given.")
    elif loss_pn is not None:
        warnings.warn(f"loss_pn was given, but it was ignored, as loss_cc_pn was also given.")

    # Return True if no errors were encountered
    return True

def default_solver(optimizer, tracked_losses, loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_pn = None, loss_ne_ov = None, loss_cc_pn = None):
    # Check validity of given arguments
    check_loss_args(loss_ne, loss_os, loss_cc, loss_ov, loss_pn, loss_ne_ov, loss_cc_pn)
    
    loss_ne = loss_ne.squeeze() if loss_ne is not None else None
    loss_os = loss_os.squeeze() if loss_os is not None else None
    loss_cc = loss_cc.squeeze() if loss_cc is not None else None
    loss_ov = loss_ov.squeeze() if loss_ov is not None else None
    loss_pn = loss_pn.squeeze() if loss_pn is not None else None
    loss_ne_ov = loss_ne_ov.squeeze() if loss_ne_ov is not None else None
    loss_cc_pn = loss_cc_pn.squeeze() if loss_cc_pn is not None else None

    if loss_ne_ov is not None and loss_cc_pn is not None:
        loss = loss_ne_ov + loss_cc_pn + loss_os
    elif loss_ne_ov is not None:
        loss = loss_ne_ov + loss_os + loss_cc + loss_pn
    elif loss_cc_pn is not None: 
        loss = loss_cc_pn + loss_ne + loss_os + loss_ov
    else:
        loss = loss_ne + loss_os + loss_cc + loss_ov + loss_pn

    # Optimisation
    loss.backward()
    optimizer.step()

    # Track individual losses
    step_ne_loss = loss_ne.detach().cpu().numpy() if loss_ne is not None else 0 # noqa
    step_os_loss = loss_os.detach().cpu().numpy() if loss_os is not None else 0 # noqa
    step_cc_loss = loss_cc.detach().cpu().numpy() if loss_cc is not None else 0 # noqa
    step_ov_loss = loss_ov.detach().cpu().numpy() if loss_ov is not None else 0 # noqa
    step_pn_loss = loss_pn.detach().cpu().numpy() if loss_pn is not None else 0 # noqa
    step_ne_ov_loss = loss_ne_ov.detach().cpu().numpy() if loss_ne_ov is not None else 0 # noqa
    step_cc_pn_loss = loss_cc_pn.detach().cpu().numpy() if loss_cc_pn is not None else 0 # noqa
    step_train_loss = loss.detach().cpu().numpy()

    tracked_losses["Nuclei Encapsulation Loss"].append(step_ne_loss)
    tracked_losses["Oversegmentation Loss"].append(step_os_loss)
    tracked_losses["Cell Calling Loss"].append(step_cc_loss)
    tracked_losses["Overlap Loss"].append(step_ov_loss)
    tracked_losses["Pos-Neg Marker Loss"].append(step_pn_loss)
    tracked_losses["Total Loss"].append(step_train_loss)
    tracked_losses["Combined Nuclei Encapsulation + Overlap Loss"].append(step_ne_ov_loss.item())
    tracked_losses["Combined Cell Calling + Marker Loss"].append(step_cc_pn_loss.item())

    return step_train_loss

def procrustes_method(model, optimizer, tracked_losses, loss_ne = None, loss_os = None, loss_cc = None, loss_ov = None, loss_pn = None, loss_ne_ov = None, loss_cc_pn = None, scale_mode = "min"): 
    # Check validity of given arguments
    check_loss_args(loss_ne, loss_os, loss_cc, loss_ov, loss_pn, loss_ne_ov, loss_cc_pn)
    
    # Track individual losses
    if loss_ne is not None:
        tracked_losses["Nuclei Encapsulation Loss"].append(loss_ne.item())
    if loss_os is not None:
        tracked_losses["Oversegmentation Loss"].append(loss_os.item())
    if loss_cc is not None:
        tracked_losses["Cell Calling Loss"].append(loss_cc.item())
    if loss_ov is not None:
        tracked_losses["Overlap Loss"].append(loss_ov.item())
    if loss_pn is not None:
        tracked_losses["Pos-Neg Marker Loss"].append(loss_pn.item())
    if loss_ne_ov is not None:
        tracked_losses["Combined Nuclei Encapsulation and Overlap Loss"].append(loss_ne_ov.item())
        #print(f"loss_ne_ov = {loss_ne_ov.item()}")
    if loss_cc_pn is not None:
        tracked_losses["Combined Cell Calling and Marker Loss"].append(loss_cc_pn.item())
        #print(f"loss_cc_pn = {loss_cc_pn.item()}")

    # Get the gradients
    loss_vals = [loss_os]
    if loss_ne_ov is not None:
        loss_vals.append(loss_ne_ov)
    else:
        loss_vals.extend([loss_ne, loss_ov])
    if loss_cc_pn is not None:
        loss_vals.append(loss_cc_pn)
    else:
        loss_vals.extend([loss_cc, loss_pn])
        
    grads = []
    for loss in loss_vals:
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward(retain_graph=True)  # Retain graph for backpropagation
        grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in model.parameters()])
        grads.append(grad)

    grads = torch.stack(grads, dim=0)  # Stack gradients

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

    if loss_ne_ov is not None and loss_cc_pn is not None:
        total_loss = loss_ne_ov + loss_cc_pn + loss_os
    elif loss_ne_ov is not None:
        total_loss = loss_ne_ov + loss_os + loss_cc + loss_pn
    elif loss_cc_pn is not None: 
        total_loss = loss_cc_pn + loss_ne + loss_os + loss_ov
    else:
        total_loss = loss_ne + loss_os + loss_cc + loss_ov + loss_pn
    
    tracked_losses["Total Loss"].append(total_loss.item())  # Track total loss

    return total_loss.item()

def plot_overlaid_losses(total_loss_vals, total_loss_ma, other_loss_vals, other_loss_ma, total_epochs, 
                         train_loader_len, use_procrustes_title, experiment_path, scale_mode=None, 
                         log_scale=True, rescaling=True, show_moving_averages=True):
    # Plots all the losses on one graph
    
    plt.figure(figsize=(18, 8))

    if rescaling:
        divisor = max(total_loss_vals) / 1000 if max(total_loss_vals) != 0 else 1
        total_loss_vals = np.divide(total_loss_vals, divisor)
    plt.plot(total_loss_vals, label="Total Loss", linewidth=1)

    divisors = {}
    for label, loss_vals in other_loss_vals.items():
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
    
        for label, loss_ma in other_loss_ma.items():
            if rescaling:
                divisor = divisors[label]
                loss_ma = np.divide(loss_ma, divisor)
            plt.plot(loss_ma, label=label, linewidth=1, alpha=0.5)

    for epoch in range(total_epochs):
        plt.axvline(x=epoch * train_loader_len, color="r", linestyle="--", alpha=0.5)

    if log_scale:
        plt.yscale("log")
                             
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    title = f"Training Loss with Procrustes Method (scaling mode: {scale_mode})" if use_procrustes_title else "Training Loss with Default Method"
    if rescaling:
        title = title + " (rescaled to max=1000)"
    plt.legend()
    plt.tight_layout()

    filename = "training_losses_overlaid.pdf" if not rescaling else "training_losses_overlaid_rescaled.pdf"
    save_path = os.path.join(experiment_path, filename)
    plt.savefig(save_path)
    #plt.show()

def plot_loss(loss_vals, ma_loss_vals, label, total_epochs, use_procrustes_title, experiment_path, train_loader_len,
              scale_mode=None, log_scale=True, rescaling=True, show_moving_averages=True):
    # Plots a single objective's values over the course of the training cycle
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
    
    if use_procrustes_title:
        title = f"{label} During Training with Procrustes Method (scaling mode: {scale_mode})"
    else: 
        title = f"{label} During Training with Default Method"

    if rescaling:
        title = title + " (rescaled to max=1000)"
    plt.title(title)
    plt.tight_layout()
    
    underscored_label = "_".join(label.lower().split(" "))
    filename = f"training_{underscored_label}.pdf" if not rescaling else f"training_{underscored_label}_rescaled.pdf"
    plt.savefig(os.path.join(experiment_path, filename))
    #plt.show()

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
    weight_mode = config.training_params.weight_mode
    ne_weight = config.training_params.ne_weight
    os_weight = config.training_params.os_weight
    cc_weight = config.training_params.cc_weight
    ov_weight = config.training_params.ov_weight
    pos_weight = config.training_params.pos_weight
    neg_weight = config.training_params.neg_weight
    
    # Loss functions
    criterion_ne = NucleiEncapsulationLoss(ne_weight, device)
    criterion_os = Oversegmentation(os_weight, device)
    criterion_cc = CellCallingLoss(cc_weight, device)
    criterion_ov = OverlapLoss(ov_weight, device)
    criterion_pn = PosNegMarkerLoss(pos_weight, neg_weight, device)

    # Whether to use static or dynamic loss weighting
    weight_mode = config.training_params.weight_mode

    # Combined loss functions if desired
    combine_losses = config.training_params.combine_losses
    combine_mode = config.training_params.combine_mode
    if combine_losses: 
        criterion_ne_ov = NucEncapOverlapLoss(ne_weight, ov_weight, device)
        criterion_cc_pn = CellCallingMarkerLoss(cc_weight, pos_weight, neg_weight, device)
    else: 
        criterion_ne_ov, criterion_cc_pn = None, None

    # Solver and learning rate
    if selected_solver is None: 
        selected_solver = config.training_params.solver
    if learning_rate is None:
        learning_rate = config.training_params.learning_rate

    # Generate path for saving outputs
    timestamp = get_experiment_id(
        make_new,
        config.experiment_dirs.dir_id,
        config.files.data_dir,
    )
    experiment_path = os.path.join(config.files.data_dir, "model_outputs", f"{timestamp}_{selected_solver}_lr-{learning_rate}")
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
    losses = {
        "Nuclei Encapsulation Loss": [],
        "Oversegmentation Loss": [],
        "Cell Calling Loss": [],
        "Overlap Loss": [],
        "Pos-Neg Marker Loss": [],
        "Combined Nuclei Encapsulation and Overlap Loss": [], 
        "Combined Cell Calling and Marker Loss": [],
        "Total Loss": [],
    }

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

    if "procrustes" in selected_solver:
        logging.info("Begin training using Procrustes method")
    else:
        logging.info("Begin training using default method")

    model = model.train()

    lrs = []
    scale_mode = ""
    
    first_step_done = False
    for epoch in range(initial_epoch, config.training_params.total_epochs):
        cur_lr = optimizer.param_groups[0]["lr"]
        print("\nEpoch =", (epoch + 1), " lr =", cur_lr)

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

            optimizer.zero_grad()

            seg_pred = model(batch_x313)

            # Compute losses
            loss_os = criterion_os(seg_pred, batch_n, os_weight)
            compute_individual_losses = not first_step_done if combine_losses else True
            if compute_individual_losses:
                loss_ne = criterion_ne(seg_pred, batch_n, ne_weight)
                loss_ov = criterion_ov(seg_pred, batch_n, ov_weight)
                loss_cc = criterion_cc(seg_pred, batch_sa, cc_weight)
                loss_pn = criterion_pn(seg_pred, batch_pos, batch_neg, pos_weight, neg_weight)

                if combine_losses:
                    logging.info(f"Computed individual losses for first step; all subsequent steps will use combined losses.")
                    if weight_mode == "dynamic":
                        # Adjust loss weights to compensate for different magnitudes of initial values
                        if combine_mode == "top":
                            ne_ov_ratio = loss_ne.item() / loss_ov.item() if loss_ov.item() != 0 else 1
                            cc_pn_ratio = loss_cc.item() / loss_pn.item() if loss_pn.item() != 0 else 1
                        else: 
                            max_ne_loss = criterion_ne.get_max(seg_pred, ne_weight)
                            max_ov_loss = criterion_ov.get_max(seg_pred, batch_n, ov_weight)
                            ne_ov_ratio = max_ne_loss / max_ov_loss if max_ov_loss != 0 else 1
                            
                            max_cc_loss = criterion_cc.get_max(seg_pred, batch_sa, cc_weight)
                            max_pn_loss = criterion_pn.get_max(seg_pred, batch_pos, batch_neg, pos_weight, neg_weight)
                            cc_pn_ratio = max_cc_loss / max_pn_loss if max_pn_loss != 0 else 1
    
                        if ne_ov_ratio != 1: 
                            ov_weight = ov_weight * ne_ov_ratio
                            logging.info(f"ne_ov_ratio={ne_ov_ratio}; ov_weight adjusted to new value of {ov_weight} to compensate.")
                        
                        if cc_pn_ratio != 1:
                            pos_weight = pos_weight * cc_pn_ratio
                            neg_weight = neg_weight * cc_pn_ratio
                            logging.info(f"cc_pn_ratio={cc_pn_ratio}; pos_weight adjusted to new value of {pos_weight} "
                                         f"and neg_weight adjusted to new value of {neg_weight} to compensate.")

                first_step_done = True
            else:
                loss_ne, loss_cc, loss_ov, loss_pn = None, None, None, None

            if combine_losses:
                loss_ne_ov = criterion_ne_ov(seg_pred, batch_n, ne_weight, ov_weight)
                loss_cc_pn = criterion_cc_pn(seg_pred, batch_sa, batch_pos, batch_neg, cc_weight, pos_weight, neg_weight)
            else: 
                loss_ne_ov, loss_cc_pn = None, None

            # Apply the Procrustes method
            if "procrustes" in selected_solver:
                scale_mode = "median" if "median" in selected_solver else "rmse" if "rmse" in selected_solver else "min"
                if combine_losses:
                    total_loss = procrustes_method(model, optimizer, losses, loss_ne_ov=loss_ne_ov, loss_os=loss_os, loss_cc_pn=loss_cc_pn, scale_mode=scale_mode)
                else:
                    total_loss = procrustes_method(model, optimizer, losses, loss_ne, loss_os, loss_cc, loss_ov, loss_pn, scale_mode=scale_mode)
            else: 
                if combine_losses:
                    total_loss = default_solver(optimizer, losses, loss_ne_ov=loss_ne_ov, loss_os=loss_os, loss_cc_pn=loss_cc_pn)
                else:
                    total_loss = default_solver(optimizer, losses, loss_ne, loss_os, loss_cc, loss_ov, loss_pn)

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

    # Calculate loss moving averages as 2.5% increments (e.g. 10 epochs x 1000 steps/epoch = 10,000 steps, i.e. 250 steps per point. 
    ma_losses = {}
    
    for loss_name, loss_vals in losses.items():
        window_width = int(len(loss_vals) / 40)
        loss_vals = np.array(loss_vals)
        
        moving_averages = []
        for i in np.arange(window_width):
            moving_averages.append(loss_vals[:i].mean())
        for i in np.arange(window_width, len(loss_vals)):
            moving_averages.append(loss_vals[i-window_width:i].mean())

        ma_losses[loss_name] = (moving_averages, window_width)

    # Plot losses
    use_procrustes_title = "procrustes" in selected_solver

    total_epochs = config.training_params.total_epochs
    train_loader_len = len(train_loader)

    # Plot all losses on one graph
    print(f"Graphing overlaid losses...")
    total_loss_vals = losses["Total Loss"]
    total_loss_ma = ma_losses["Total Loss"]
    if combine_losses:
        keys = ["Combined Nuclei Encapsulation and Overlap Loss", "Combined Cell Calling and Marker Loss", "Oversegmentation Loss"]
    else: 
        keys = ["Nuclei Encapsulation Loss", "Oversegmentation Loss", "Cell Calling Loss", "Overlap Loss", "Pos-Neg Marker Loss"]
    other_loss_vals = {key:losses[key] for key in keys}
    other_loss_ma = {key:ma_losses[key] for key in keys}
    plot_overlaid_losses(total_loss_vals, total_loss_ma, other_loss_vals, other_loss_ma, total_epochs, 
                         train_loader_len, use_procrustes_title, experiment_path, scale_mode, 
                         log_scale=True, rescaling=False)

    # Plot individual losses
    print(f"Graphing total loss...")
    plot_loss(losses["Total Loss"], ma_losses["Total Loss"], "Total Loss", total_epochs, use_procrustes_title, experiment_path, 
              train_loader_len, scale_mode, log_scale=True, rescaling=False)
    print(f"Graphing individual losses...")
    for key in keys:
        plot_loss(losses[key], ma_losses[key], key, total_epochs, use_procrustes_title, experiment_path, 
                  train_loader_len, scale_mode, log_scale=True, rescaling=False)

    # Repeat for rescaled versions
    print(f"Graphing overlaid rescaled losses...")
    plot_overlaid_losses(total_loss_vals, total_loss_ma, other_loss_vals, other_loss_ma, total_epochs, 
                         train_loader_len, use_procrustes_title, experiment_path, scale_mode, 
                         log_scale=True, rescaling=True)
    print(f"Graphing rescaled total loss...")
    plot_loss(losses["Total Loss"], ma_losses["Total Loss"], "Total Loss", total_epochs, use_procrustes_title, experiment_path, 
              train_loader_len, scale_mode, log_scale=True, rescaling=True)
    print(f"Graphing rescaled individual losses...")
    for key in keys:
        plot_loss(losses[key], ma_losses[key], key, total_epochs, use_procrustes_title, experiment_path, 
                  train_loader_len, scale_mode, log_scale=True, rescaling=True)

    logging.info("Training finished")

    return losses, ma_losses, experiment_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")

    args = parser.parse_args()
    config = load_config(args.config_dir)

    train(config)
