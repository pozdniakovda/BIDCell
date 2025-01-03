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

from .solvers.solvers import default_solver, procrustes_method

from .data_vis.plot_losses import (
    plot_overlaid_losses, 
    plot_losses, 
    get_ma_losses,
    get_solver_title
)
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

def get_weighting_ratio(loss1, loss2, weights1, weights2, weights1_names, weights2_names, input_shape):
    loss1 = to_scalar(loss1)
    loss2 = to_scalar(loss2)
    ratio = None
    
    logging.info(f"Dynamically adjusting weights based on loss values at first step.")
    if loss1 != 0 and loss2 != 0:
        ratio = loss1 / loss2
        weights2 = [weight2 * ratio for weight2 in weights2]
        message = f"loss1={loss1}, loss2={loss2}, ratio={ratio}"
        for weight_name, weight_val in zip(weights2_names, weights2):
            message = f"{message}; {weight_name} adjusted to new value of {weight_val}"
        logging.info(message)
    
    return ratio, weights1, weights2

def compute_individual_losses(seg_pred, batch_n, batch_sa, batch_pos, batch_neg, batch_expr_sum, weights, device,
                              combine_ne_ov=False, combine_os_ov=False, combine_cc_pn=False, is_first_step=True):
    # Compute individual losses as appropriate
    loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn = None, None, None, None, None, None

    # Loss functions
    criterion_ne = NucleiEncapsulationLoss(weights["ne"], device)
    criterion_os = OversegmentationLoss(weights["os"], device)
    criterion_cc = CellCallingLoss(weights["cc"], device)
    criterion_ov = OverlapLoss(weights["ov"], device)
    criterion_mu = MultipleAssignmentLoss(weights["mu"], device)
    criterion_pn = PosNegMarkerLoss(weights["pos"], weights["neg"], device)

    # Apply losses
    ov_combined = combine_ne_ov or combine_os_ov
    if is_first_step or not ov_combined:
        loss_ov = criterion_ov(seg_pred, batch_n, weights["ov"])
    
    if is_first_step or not combine_ne_ov:
        loss_ne = criterion_ne(seg_pred, batch_n, weights["ne"])
    if is_first_step or not combine_os_ov:
        loss_os = criterion_os(seg_pred, batch_n, weights["os"])
    if is_first_step or not combine_cc_pn:
        loss_cc = criterion_cc(seg_pred, batch_sa, weights["cc"])
        loss_pn = criterion_pn(seg_pred, batch_pos, batch_neg, weights["pos"], weights["neg"])
    
    loss_mu = criterion_mu(seg_pred, batch_expr_sum, weights["mu"])

    return (loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn)

def compute_losses(seg_pred, batch_n, batch_sa, batch_pos, batch_neg, batch_expr_sum, weights, device,
                   combine_ne_ov=False, combine_os_ov=False, combine_cc_pn=False, is_first_step=True):
    '''
    Computes losses from input tensors

    Args: 
        seg_pred:        predicted segmentations
        batch_n:         nuclei map masked by search areas;       shape: [H, W, n_cells]
        batch_sa:        search areas defining individual cells;  shape: [H, W, n_cells]
        batch_pos:       positive markers masked by search areas; shape: [H, W, n_cells]
        batch_neg:       negative markers masked by search areas; shape: [H, W, n_cells]
        coords_h1:       coords_h1
        coords_w1:       coords_w1
        nucl_aug:        augmented nuclei map;                    shape: [H, W]
        batch_expr_sum:  summed expression map;                   shape: [H, W]

    Returns: 
        loss_ne:         nuclei encapsulation loss
        loss_os:         oversegmentation loss
        loss_cc:         cell calling loss
        loss_ov:         overlap loss
        loss_mu:         multiple assignment loss
        loss_pn:         positive/negative marker loss
        loss_ne_ov:      combined loss_ne + loss_ov
        loss_os_ov:      combined loss_os + loss_ov
        loss_cc_pn:      combined loss_cc + loss_pn
        weights:         weights
    '''
    
    # Compute individual losses as appropriate
    individual_losses = compute_individual_losses(seg_pred, batch_n, batch_sa, batch_pos, batch_neg, batch_expr_sum, weights,
                                                  device, combine_ne_ov, combine_os_ov, combine_cc_pn, is_first_step)
    loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn = individual_losses
    loss_ne_ov, loss_os_ov, loss_cc_pn = None, None, None
    
    if is_first_step:
        if combine_ne_ov or combine_os_ov or combine_cc_pn:
            logging.info(f"Computing all constituent losses for first step; subsequent steps will use combined losses.")

    # If selected, adjust loss weights to compensate for different magnitudes of initial values
    if combine_ne_ov:
        if is_first_step:
            ratio, weights1, weights2 = get_weighting_ratio(loss_ne, loss_ov, [weights["ne"]], [ov_weight], 
                                                            ["ne_weight"], ["ov_weight"], seg_pred.shape)
            weights["ne"] = weights1[0]
            weights["ov"] = weights2[0]
            loss_ne, loss_ov = None, None # reset to make sure they are not used in grads
        criterion_ne_ov = NucEncapOverlapLoss(weights["ne"], weights["ov"], device) if combine_ne_ov else None
        loss_ne_ov = criterion_ne_ov(seg_pred, batch_n, weights["ne"], weights["ov"]) if combine_ne_ov else None

    if combine_os_ov:
        if is_first_step:
            ratio, weights1, weights2 = get_weighting_ratio(loss_os, loss_ov, [weights["os"]], [weights["ov"]], 
                                                            ["os_weight"], ["ov_weight"], seg_pred.shape)
            weights["os"] = weights1[0]
            weights["ov"] = weights2[0]
            loss_os, loss_ov = None, None # reset to make sure they are not used in grads
        criterion_os_ov = OversegOverlapLoss(weights["os"], weights["ov"], device) if combine_os_ov else None
        loss_os_ov = criterion_os_ov(seg_pred, batch_n, weights["os"], weights["ov"]) if combine_os_ov else None    
        
    if combine_cc_pn:
        if is_first_step:
            ratio, weights1, weights2 = get_weighting_ratio(loss_cc, loss_pn, [weights["cc"]], [weights["pos"], weights["neg"]], 
                                                            ["cc_weight"], ["pos_weight", "neg_weight"], seg_pred.shape)
            weights["cc"] = weights1[0]
            weights["pos"] = weights2[0]
            weights["neg"] = weights2[1]
            loss_cc, loss_pn = None, None # reset to make sure they are not used in grads
        criterion_cc_pn = CellCallingMarkerLoss(weights["cc"], weights["pos"], weights["neg"], device) if combine_cc_pn else None
        loss_cc_pn = criterion_cc_pn(seg_pred, batch_sa, batch_pos, batch_neg, weights["cc"], weights["pos"], weights["neg"]) if combine_cc_pn else None
    
    is_first_step = False # first step special case handling is concluded
    
    return (loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, loss_ne_ov, loss_os_ov, loss_cc_pn, weights)

def generate_paths(config, make_new, learning_rate, dynamic_solvers, selected_solver=None, 
                   starting_solver=None, ending_solver=None, epochs_before_switch=0):
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
    
    return experiment_path

def get_scheduler(total_epochs, optimizer, global_step):
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    
    lf = (
        lambda x: (
            ((1 + math.cos(x * math.pi / total_epochs)) / 2)
            ** 1.0
        )
        * 0.95
        + 0.05
    )  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = global_step
    
    return scheduler

def get_optimizer(config, model, learning_rate):
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

    return optimizer

def detach_fig_outputs(coords_h1, coords_w1, seg_pred, nucl_aug, batch_sa, expr_aug_sum):
    # Detaches fig outputs from device and returns them so they can be saved
    
    coords_h1 = coords_h1.detach().cpu().squeeze().numpy()
    coords_w1 = coords_w1.detach().cpu().squeeze().numpy()
    sample_seg = seg_pred.detach().cpu().numpy()
    sample_n = nucl_aug.detach().cpu().numpy()
    sample_sa = batch_sa.detach().cpu().numpy()
    sample_expr = expr_aug_sum.detach().cpu().numpy()

    return (coords_h1, coords_w1, sample_seg, sample_n, sample_sa, sample_expr)

def save_model(config, experiment_path, epoch, step_epoch, model, optimizer):
    # Save model
    save_path = os.path.join(experiment_path, config.experiment_dirs.model_dir, 
                             f"epoch_{epoch+1}_step_{step_epoch}.pth")
    output_dict = {"epoch": epoch + 1,
                   "model_state_dict": model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict()}
    
    torch.save(output_dict, save_path)
    logging.info("Model saved: %s" % save_path)

def restore_saved_model(config, experiment_path, resume_epoch, resume_step, optimizer):
    # Restore saved model
    load_path = os.path.join(experiment_path, config.experiment_dirs.model_dir, 
                             f"epoch_{resume_epoch}_step_{resume_step}.pth")
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    assert epoch == resume_epoch
    print("Resume training, successfully loaded " + load_path)

    return model, optimizer, epoch

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
    make_new = True if resume_epoch is None else False

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
    weights = {"ne": config.training_params.ne_weight, 
               "os": config.training_params.os_weight, 
               "cc": config.training_params.cc_weight, 
               "ov": config.training_params.ov_weight, 
               "mu": config.training_params.mu_weight, 
               "pos": config.training_params.pos_weight, 
               "neg": config.training_params.neg_weight}

    # Overlap loss preferences
    ov_distance_scaling = config.training_params.ov_distance_scaling
    ov_intensity_weighting = config.training_params.ov_intensity_weighting
    
    # Combined loss functions if desired
    combine_ne_ov = config.training_params.combine_ne_ov
    combine_os_ov = config.training_params.combine_os_ov
    combine_cc_pn = config.training_params.combine_cc_pn
    if combine_ne_ov and combine_os_ov: 
        raise Exception(f"combine_ne_ov and combine_os_ov were both set to True, but they are "
                        "mutually exclusive because they both use OverlapLoss.")

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
    experiment_path = generate_paths(config, make_new, learning_rate, dynamic_solvers, selected_solver, 
                                     starting_solver, ending_solver, epochs_before_switch)

    # Optimiser
    optimizer = get_optimizer(config, model, learning_rate)

    global_step = 0
    losses = {}

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    scheduler = get_scheduler(config.training_params.total_epochs, optimizer, global_step)

    # Starting epoch
    initial_epoch = resume_epoch if resume_epoch is not None else 0

    # Restore saved model
    if resume_epoch is not None:
        model, optimizer, epoch = restore_saved_model(config, experiment_path, resume_epoch, resume_step, optimizer)

    if dynamic_solvers:
        logging.info(f"Begin training using {starting_solver} for {epochs_before_switch} epochs, followed by {ending_solver} thereafter")
    elif "procrustes" in selected_solver:
        logging.info("Begin training using Procrustes method")
        scale_mode = "median" if "median" in selected_solver else "rmse" if "rmse" in selected_solver else "min"        
    else:
        logging.info("Begin training using default method")

    model = model.train()

    lrs = []
    scale_mode = ""

    total_epochs = config.training_params.total_epochs
    is_first_step = True
    for epoch in range(initial_epoch, total_epochs):
        # Define current solver
        if dynamic_solvers:
            current_solver = starting_solver if epoch < epochs_before_switch else ending_solver
        else:
            current_solver = selected_solver

        cur_lr = optimizer.param_groups[0]["lr"]
        print("\nEpoch =", (epoch + 1), " lr =", cur_lr, " solver =", current_solver)

        for step_epoch, (
            batch_ess,       # shape: [H, W, n_cells]
            batch_x313,      # shape: [H, W, n_channels, n_cells]
            batch_n,         # shape: [H, W, n_cells]
            batch_sa,        # shape: [H, W, n_cells]
            batch_pos,       # shape: [H, W, n_cells]
            batch_neg,       # shape: [H, W, n_cells]
            coords_h1,
            coords_w1,
            nucl_aug,        # shape: [H, W]
            batch_expr_sum,  # shape: [H, W]
        ) in enumerate(train_loader): 
            # Permute channels axis to batch axis
            batch_ess = batch_ess.permute(3, 0, 1, 2)                  # new shape: [n_cells, 1, H, W]
            batch_x313 = batch_x313[0, :, :, :, :].permute(3, 2, 0, 1) # new shape: [n_cells, n_channels, H, W]
            batch_n = batch_n.permute(3, 0, 1, 2)                      # new shape: [n_cells, 1, H, W]
            batch_sa = batch_sa.permute(3, 0, 1, 2)                    # new shape: [n_cells, 1, H, W]
            batch_pos = batch_pos.permute(3, 0, 1, 2)                  # new shape: [n_cells, 1, H, W]
            batch_neg = batch_neg.permute(3, 0, 1, 2)                  # new shape: [n_cells, 1, H, W]
            batch_expr_sum = batch_expr_sum.unsqueeze(0)               # new shape: [1, 1, H, W]

            if batch_x313.shape[0] == 0:
                # Save the model periodically
                if (step_epoch % model_freq) == 0:
                    filename = f"epoch_{epoch+1}_step_{step_epoch}.pth"
                    save_path = os.path.join(experiment_path, config.experiment_dirs.model_dir, filename)
                    output_dict = {"epoch": epoch + 1,
                                   "model_state_dict": model.state_dict(),
                                   "optimizer_state_dict": optimizer.state_dict()}
                    torch.save(output_dict, save_path)
                    logging.info("Model saved: %s" % save_path)
                
                continue

            print(f"Shapes of tensors immediately before being transferred to GPU: \n"
                  f"\tbatch_ess shape: {batch_ess.shape}\n"
                  f"\tbatch_x313 shape: {batch_x313.shape}\n"
                  f"\tbatch_n shape: {batch_n.shape}\n"
                  f"\tbatch_sa shape: {batch_sa.shape}\n"
                  f"\tbatch_pos shape: {batch_pos.shape}\n"
                  f"\tbatch_neg shape: {batch_neg.shape}\n"
                  f"\texpr_aug_sum (batch_expr_sum) shape: {batch_expr_sum.shape}")
            
            # Transfer to GPU
            batch_ess = batch_ess.to(device)
            batch_x313 = batch_x313.to(device)
            batch_sa = batch_sa.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)
            batch_n = batch_n.to(device)
            batch_expr_sum = batch_expr_sum.to(device)

            optimizer.zero_grad()

            seg_pred = model(batch_x313) # binary prediction of cell or not; shape: [n_cells, 2, H, W]
            print(f"\tseg_pred shape: {seg_pred.shape}")

            # Compute individual losses as appropriate
            computed_losses = compute_losses(seg_pred, batch_n, batch_sa, batch_pos, batch_neg, batch_expr_sum, 
                                             weights, device, combine_ne_ov, combine_os_ov, combine_cc_pn, is_first_step)
            loss_ne, loss_os, loss_cc, loss_ov, loss_mu, loss_pn, loss_ne_ov, loss_os_ov, loss_cc_pn, weights = computed_losses
            
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
                fig_outputs = detach_fig_outputs(coords_h1, coords_w1, seg_pred, 
                                                 nucl_aug, batch_sa, expr_aug_sum)
                coords_h1, coords_w1, sample_seg, sample_n, sample_sa, sample_expr = fig_outputs
                patch_fp = os.path.join(f"{experiment_path}/{config.experiment_dirs.samples_dir}", 
                                        f"epoch_{epoch+1}_{step_epoch}_{coords_h1}_{coords_w1}.png")
                save_fig_outputs(sample_seg, sample_n, sample_sa, sample_expr, patch_fp)
                
                print(f"Epoch[{epoch+1}/{total_epochs}], Step[{step_epoch}], Total Loss:{total_loss:.4f}")

            # Save model
            if (step_epoch % model_freq) == 0:
                save_model(config, experiment_path, epoch, step_epoch, model, optimizer)

            global_step += 1

        # Update and append current LR
        scheduler.step()
        lrs.append(cur_lr)

    # Graph the losses
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
