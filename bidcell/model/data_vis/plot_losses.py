import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from ...config import load_config, Config

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

def to_scalar(value):
    # Helper function that converts one-item Torch tensors into Python scalars (e.g. float)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.item()
        else:
            print("Cannot apply .item() to a tensor with more than one element.")
    return value

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
    plt.grid(True, axis="y")
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
            plt.grid(True, axis="y")
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

