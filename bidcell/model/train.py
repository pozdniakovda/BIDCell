import argparse
import logging
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
)
from .model.model import SegmentationModel as Network
from .utils.utils import (
    get_experiment_id,
    make_dir,
    save_fig_outputs,
)
from ..config import load_config, Config

def default_solver(loss_ne, loss_os, loss_cc, loss_ov, loss_pn, optimizer, tracked_losses):
    loss_ne = loss_ne.squeeze()
    loss_os = loss_os.squeeze()
    loss_cc = loss_cc.squeeze()
    loss_ov = loss_ov.squeeze()
    loss_pn = loss_pn.squeeze()

    loss = loss_ne + loss_os + loss_cc + loss_ov + loss_pn

    # Optimisation
    loss.backward()
    optimizer.step()

    # Track individual losses
    step_ne_loss = loss_ne.detach().cpu().numpy() # noqa
    step_os_loss = loss_os.detach().cpu().numpy() # noqa
    step_cc_loss = loss_cc.detach().cpu().numpy() # noqa
    step_ov_loss = loss_ov.detach().cpu().numpy() # noqa
    step_pn_loss = loss_pn.detach().cpu().numpy() # noqa
    step_train_loss = loss.detach().cpu().numpy()

    tracked_losses["Nuclei Encapsulation Loss"].append(step_ne_loss)
    tracked_losses["Oversegmentation Loss"].append(step_os_loss)
    tracked_losses["Cell Calling Loss"].append(step_cc_loss)
    tracked_losses["Overlap Loss"].append(step_ov_loss)
    tracked_losses["Pos-Neg Marker Loss"].append(step_pn_loss)
    tracked_losses["Total Loss"].append(step_train_loss)

    return step_train_loss

def procrustes_method(loss_ne, loss_os, loss_cc, loss_ov, loss_pn, model, optimizer, tracked_losses, scale_mode = "min"): 
    # Track individual losses
    tracked_losses["Nuclei Encapsulation Loss"].append(loss_ne.item())
    tracked_losses["Oversegmentation Loss"].append(loss_os.item())
    tracked_losses["Cell Calling Loss"].append(loss_cc.item())
    tracked_losses["Overlap Loss"].append(loss_ov.item())
    tracked_losses["Pos-Neg Marker Loss"].append(loss_pn.item())

    grads = []
    for loss in [loss_ne, loss_os, loss_cc, loss_ov, loss_pn]:
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

    total_loss = loss_ne + loss_os + loss_cc + loss_ov + loss_pn
    tracked_losses["Total Loss"].append(total_loss.item())  # Track total loss

    return total_loss.item()

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

    # Loss functions
    criterion_ne = NucleiEncapsulationLoss(config.training_params.ne_weight, device)
    criterion_os = Oversegmentation(config.training_params.os_weight, device)
    criterion_cc = CellCallingLoss(config.training_params.cc_weight, device)
    criterion_ov = OverlapLoss(config.training_params.ov_weight, device)
    criterion_pn = PosNegMarkerLoss(
        config.training_params.pos_weight,
        config.training_params.neg_weight,
        device,
    )

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
            loss_ne = criterion_ne(seg_pred, batch_n)
            loss_os = criterion_os(seg_pred, batch_n)
            loss_cc = criterion_cc(seg_pred, batch_sa)
            loss_ov = criterion_ov(seg_pred, batch_n)
            loss_pn = criterion_pn(seg_pred, batch_pos, batch_neg)

            # Apply the Procrustes method
            if "procrustes" in selected_solver:
                scale_mode = "median" if "median" in selected_solver else "rmse" if "rmse" in selected_solver else "min"
                total_loss = procrustes_method(loss_ne, loss_os, loss_cc, loss_ov, loss_pn, model, optimizer, losses, scale_mode)
            else: 
                total_loss = default_solver(loss_ne, loss_os, loss_cc, loss_ov, loss_pn, optimizer, losses)

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
    
    plt.figure(figsize=(18, 8))

    plt.plot(losses["Total Loss"], label="Total Loss", linewidth=1)

    plt.plot(losses["Nuclei Encapsulation Loss"], label="Nuclei Encapsulation Loss", linewidth=0.5)
    plt.plot(losses["Oversegmentation Loss"], label="Oversegmentation Loss", linewidth=0.5)
    plt.plot(losses["Cell Calling Loss"], label="Cell Calling Loss", linewidth=0.5)
    plt.plot(losses["Overlap Loss"], label="Overlap Loss", linewidth=0.5)
    plt.plot(losses["Pos-Neg Marker Loss"], label="Pos-Neg Marker Loss", linewidth=0.5)

    ma_loss_vals, ma_window_width = ma_losses["Total Loss"]
    plt.plot(ma_loss_vals, label=f"Total Loss (moving average, {ma_window_width})", linewidth=2)

    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    if use_procrustes_title:
        plt.title(f"Training Loss with Procrustes Method (scaling mode: {scale_mode})")
    else:
        plt.title("Training Loss with Default Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_losses.pdf"))
    #plt.show()

    # Plot individual losses
    plt.figure(figsize=(18, 8))
    plt.plot(losses["Total Loss"], label="Total Loss", linewidth=0.5)
    ma_loss_vals, ma_window_width = ma_losses["Total Loss"]
    plt.plot(ma_loss_vals, label=f"Total Loss (moving average, {ma_window_width})", linewidth=2)
    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    if use_procrustes_title:
        plt.title(f"Total Loss During Training with Procrustes Method (scaling mode: {scale_mode})")
    else: 
        plt.title("Total Loss During Training with Default Method")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_total_losses.pdf"))
    #plt.show()

    plt.figure(figsize=(18, 8))
    plt.plot(losses["Nuclei Encapsulation Loss"], label="Nuclei Encapsulation Loss", linewidth=0.5, alpha=0.5)
    ma_loss_vals, ma_window_width = ma_losses["Nuclei Encapsulation Loss"]
    plt.plot(ma_loss_vals, label=f"Nuclei Encapsulation Loss (moving average, {ma_window_width})", linewidth=2)
    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    if use_procrustes_title:
        plt.title(f"Nuclei Encapsulation Loss During Training with Procrustes Method (scaling mode: {scale_mode})")
    else: 
        plt.title("Nuclei Encapsulation Loss During Training with Default Method")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_ne_losses.pdf"))
    #plt.show()

    plt.figure(figsize=(18, 8))
    plt.plot(losses["Oversegmentation Loss"], label="Oversegmentation Loss", linewidth=0.5, alpha=0.5)
    ma_loss_vals, ma_window_width = ma_losses["Oversegmentation Loss"]
    plt.plot(ma_loss_vals, label=f"Oversegmentation Loss (moving average, {ma_window_width})", linewidth=2)
    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    if use_procrustes_title:
        plt.title("Oversegmentation Loss During Training with Procrustes Method")
    else: 
        plt.title("Oversegmentation Loss During Training with Default Method")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_os_losses.pdf"))
    #plt.show()

    plt.figure(figsize=(18, 8))
    plt.plot(losses["Cell Calling Loss"], label="Cell Calling Loss", linewidth=0.5, alpha=0.5)
    ma_loss_vals, ma_window_width = ma_losses["Cell Calling Loss"]
    plt.plot(ma_loss_vals, label=f"Cell Calling Loss (moving average, {ma_window_width})", linewidth=2)
    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    if use_procrustes_title:
        plt.title(f"Cell Calling Loss During Training with Procrustes Method (scaling mode: {scale_mode})")
    else: 
        plt.title("Cell Calling Loss During Training with Default Method")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_cc_losses.pdf"))
    #plt.show()

    plt.figure(figsize=(18, 8))
    plt.plot(losses["Overlap Loss"], label="Overlap Loss", linewidth=0.5, alpha=0.5)
    ma_loss_vals, ma_window_width = ma_losses["Overlap Loss"]
    plt.plot(ma_loss_vals, label=f"Overlap Loss (moving average, {ma_window_width})", linewidth=2)
    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    if use_procrustes_title:
        plt.title(f"Overlap Loss During Training with Procrustes Method (scaling mode: {scale_mode})")
    else:
        plt.title("Overlap Loss During Training with Default Method")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_ov_losses.pdf"))
    #plt.show()

    plt.figure(figsize=(18, 8))
    plt.plot(losses["Pos-Neg Marker Loss"], label="Pos-Neg Marker Loss", linewidth=0.5, alpha=0.5)
    ma_loss_vals, ma_window_width = ma_losses["Pos-Neg Marker Loss"]
    plt.plot(ma_loss_vals, label=f"Pos-Neg Marker Loss (moving average, {ma_window_width})", linewidth=2)
    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    if use_procrustes_title:
        plt.title(f"Positive/Negative Marker Loss During Training with Procrustes Method (scaling mode: {scale_mode})")
    else:
        plt.title("Positive/Negative Marker Loss During Training with Default Method")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_pn_losses.pdf"))
    #plt.show()

    logging.info("Training finished")

    return losses, ma_losses, experiment_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")

    args = parser.parse_args()
    config = load_config(args.config_dir)

    train(config)
