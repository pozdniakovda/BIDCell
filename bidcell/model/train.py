import argparse
import logging
import math
import sys
import os
import numpy as np

import matplotlib.pyplot as plt
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


def train(config: Config):
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

    timestamp = get_experiment_id(
        make_new,
        config.experiment_dirs.dir_id,
        config.files.data_dir,
    )
    experiment_path = os.path.join(config.files.data_dir, "model_outputs", timestamp)
    make_dir(experiment_path + "/" + config.experiment_dirs.model_dir)
    make_dir(experiment_path + "/" + config.experiment_dirs.samples_dir)

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

    # Optimiser
    if config.training_params.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=config.training_params.learning_rate,
            weight_decay=1e-8,
        )
    elif config.training_params.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training_params.learning_rate,
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

    logging.info("Begin training")

    model = model.train()

    lrs = []

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

            # Track individual losses
            losses["Nuclei Encapsulation Loss"].append(loss_ne.item())
            losses["Oversegmentation Loss"].append(loss_os.item())
            losses["Cell Calling Loss"].append(loss_cc.item())
            losses["Overlap Loss"].append(loss_ov.item())
            losses["Pos-Neg Marker Loss"].append(loss_pn.item())

            grads = []
            for loss in [loss_ne, loss_os, loss_cc, loss_ov, loss_pn]:
                optimizer.zero_grad()  # Clear previous gradients
                loss.backward(retain_graph=True)  # Retain graph for backpropagation
                grad = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in model.parameters()])
                grads.append(grad)

            grads = torch.stack(grads, dim=0)  # Stack gradients

            # Apply Procrustes Solver
            grads, weights, singulars = ProcrustesSolver.apply(grads.T.unsqueeze(0))
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
            losses["Total Loss"].append(total_loss.item())  # Track total loss

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
                        total_loss.item(),
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

    # Plot losses
    plt.figure(figsize=(12, 8))

    plt.plot(losses["Total Loss"], label="Total Loss")

    plt.plot(losses["Nuclei Encapsulation Loss"], label="Nuclei Encapsulation Loss")
    plt.plot(losses["Oversegmentation Loss"], label="Oversegmentation Loss")
    plt.plot(losses["Cell Calling Loss"], label="Cell Calling Loss")
    plt.plot(losses["Overlap Loss"], label="Overlap Loss")
    plt.plot(losses["Pos-Neg Marker Loss"], label="Pos-Neg Marker Loss")

    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_losses.pdf"))
    plt.show()

    # Plot individual losses
    plt.figure(figsize=(12, 8))
    plt.plot(losses["Total Loss"], label="Total Loss")
    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Total Loss During Training with Procrustes Method")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_total_losses.pdf"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(losses["Nuclei Encapsulation Loss"], label="Nuclei Encapsulation Loss")
    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Nuclei Encapsulation Loss During Training with Procrustes Method")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_os_losses.pdf"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(losses["Oversegmentation Loss"], label="Oversegmentation Loss")
    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Oversegmentation Loss During Training with Procrustes Method")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_ne_losses.pdf"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(losses["Cell Calling Loss"], label="Cell Calling Loss")
    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Cell Calling Loss During Training with Procrustes Method")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_cc_losses.pdf"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(losses["Overlap Loss"], label="Overlap Loss")
    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Overlap Loss During Training with Procrustes Method")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_ov_losses.pdf"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(losses["Pos-Neg Marker Loss"], label="Pos-Neg Marker Loss")
    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Positive/Negative Marker Loss During Training with Procrustes Method")
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "training_pn_losses.pdf"))
    plt.show()

    # Plot scaled losses
    plt.figure(figsize=(12, 8))

    scaled_total_loss = np.array(losses["Total Loss"])
    scaled_total_loss = scaled_total_loss / scaled_total_loss.max() if scaled_total_loss.max() > 0 else scaled_total_loss
    plt.plot(scaled_total_loss, label="Scaled Total Loss")

    scaled_ne_loss = np.array(losses["Nuclei Encapsulation Loss"])
    scaled_ne_loss = scaled_ne_loss / scaled_ne_loss.max() if scaled_ne_loss.max() > 0 else scaled_ne_loss
    plt.plot(scaled_ne_loss, label="Scaled Nuclei Encapsulation Loss")

    scaled_os_loss = np.array(losses["Oversegmentation Loss"])
    scaled_os_loss = scaled_os_loss / scaled_os_loss.max() if scaled_os_loss.max() > 0 else scaled_os_loss
    plt.plot(scaled_os_loss, label="Scaled Oversegmentation Loss")

    scaled_cc_loss = np.array(losses["Cell Calling Loss"])
    scaled_cc_loss = scaled_cc_loss / scaled_cc_loss.max() if scaled_cc_loss.max() > 0 else scaled_cc_loss
    plt.plot(scaled_cc_loss, label="Scaled Cell Calling Loss")

    scaled_ov_loss = np.array(losses["Overlap Loss"])
    scaled_ov_loss = scaled_ov_loss / scaled_ov_loss.max() if scaled_ov_loss.max() > 0 else scaled_ov_loss
    plt.plot(scaled_ov_loss, label="Scaled Overlap Loss")

    scaled_pn_loss = np.array(losses["Pos-Neg Marker Loss"])
    scaled_pn_loss = scaled_pn_loss / scaled_pn_loss.max() if scaled_pn_loss.max() > 0 else scaled_pn_loss
    plt.plot(scaled_ne_loss, label="Scaled Pos-Neg Marker Loss")

    for epoch in range(config.training_params.total_epochs):
        plt.axvline(x=epoch * len(train_loader), color="r", linestyle="--", alpha=0.5)
    
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "rescaled_training_losses.pdf"))
    plt.show()

    logging.info("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")

    args = parser.parse_args()
    config = load_config(args.config_dir)

    train(config)
