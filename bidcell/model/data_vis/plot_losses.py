import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
from ...config import load_config, Config

def get_solver_title(selected_solver = None, starting_solver = None, ending_solver = None, epochs_before_switch = 0, dynamic_solvers = False):
    # Generates a title fragment referencing the solver(s) that were used during training
    
    if dynamic_solvers:        
        if "procrustes" in starting_solver.lower():
            starting_scale_mode = "median" if "median" in starting_solver else "rmse" if "rmse" in starting_solver else "min"
            starting_solver_title = f"Procrustes Method (scaling mode: {starting_scale_mode})"
        elif "stch" in starting_solver.lower():
            starting_solver_title = f"STCH-MTL Method"
        else:
            starting_solver_title = f"Default Method"

        if "procrustes" in ending_solver.lower():
            ending_scale_mode = "median" if "median" in ending_solver else "rmse" if "rmse" in ending_solver else "min"
            ending_solver_title = f"Procrustes Method (scaling mode: {ending_scale_mode})"
        elif "stch" in ending_solver.lower():
            ending_solver_title = f"STCH-MTL Method"
        else:
            ending_solver_title = f"Default Method"

        solver_title = f"{starting_solver_title} (epochs 1-{epochs_before_switch}) to {ending_solver_title} (epochs {epochs_before_switch+1} onwards)"
        
    elif "procrustes" in selected_solver:
        scale_mode = "median" if "median" in selected_solver else "rmse" if "rmse" in selected_solver else "min"
        solver_title = f"Procrustes Method (scaling mode: {scale_mode})"

    elif "stch" in selected_solver.lower():
        solver_title = f"STCH-MTL Method"
        
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
                         experiment_path, solver_title, switch_after=0, log_scale=False, 
                         rescaling=True, show_moving_averages=True):
    # Plots all the losses on one graph

    plt.figure(figsize=(18, 8))

    if rescaling:
        last_epoch_steps = int(len(total_loss_vals) / total_epochs)
        last_epoch_vals = total_loss_vals[-last_epoch_steps:]
        last_epoch_mean = sum(last_epoch_vals) / len(last_epoch_vals)
        divisor = last_epoch_mean if last_epoch_mean != 0 else 1
        total_loss_vals = np.divide(total_loss_vals, divisor) * 1000
    
    total_loss_linewidth = 1 if show_moving_averages else 2
    plt.plot(total_loss_vals, label="Total Loss", linewidth=1)

    divisors = {}
    loss_vals_count = 0
    loss_linewidth = 0.25 if show_moving_averages else 2
    loss_alpha = 0.5 if show_moving_averages else 1.0
    for label, loss_vals in other_loss_vals.items():
        if loss_vals is not None:
            loss_vals_count = max(len(loss_vals), loss_vals_count)
            if len(loss_vals) > 0:
                last_epoch_steps = int(len(loss_vals) / total_epochs)
                last_epoch_vals = loss_vals[-last_epoch_steps:]
                last_epoch_mean = sum(last_epoch_vals) / len(last_epoch_vals)
                divisor = last_epoch_mean if last_epoch_mean != 0 else 1
                if rescaling:
                    divisors[label] = divisor
                    loss_vals = np.divide(loss_vals, divisor) * 1000
                plt.plot(loss_vals, label=label, linewidth=loss_linewidth, alpha=loss_alpha)
    
    if show_moving_averages and total_loss_ma is not None:
        ma_loss_vals, ma_window_width = total_loss_ma
        if rescaling:
            last_epoch_steps = int(len(ma_loss_vals) / total_epochs)
            last_epoch_ma_vals = ma_loss_vals[-last_epoch_steps:]
            last_epoch_ma_mean = sum(last_epoch_ma_vals) / len(last_epoch_ma_vals)
            divisor = last_epoch_ma_mean if last_epoch_ma_mean != 0 else 1
            ma_loss_vals = np.divide(ma_loss_vals, divisor) * 1000
        plt.plot(ma_loss_vals, label=f"Total Loss (moving average, {ma_window_width})", linewidth=2)

        if other_loss_ma is not None:
            for label, loss_ma_tuple in other_loss_ma.items():
                loss_ma, ma_window_width = loss_ma_tuple
                if loss_ma is not None:
                    if len(loss_ma) > 0:
                        if rescaling:
                            divisor = divisors[label]
                            loss_ma = np.divide(loss_ma, divisor) * 1000
                        plt.plot(loss_ma, label=f"{label} (moving average, {ma_window_width})", linewidth=1, alpha=0.5)
    
    elif show_moving_averages:
        warnings.warn(f"Could not show moving averages because total_loss_ma is {total_loss_ma}")

    vals_per_epoch = round(loss_vals_count / total_epochs)
    vline_width = 15 / total_epochs # 1.5 for 10 epochs
    for epoch in np.arange(0, total_epochs + 1):
        color = "r" if epoch == switch_after-1 else "black"
        plt.axvline(x=epoch*vals_per_epoch - 1, color=color, linewidth=vline_width, linestyle="--", alpha=0.5)

    if log_scale:
        plt.yscale("log")
    else:
        plt.ylim(bottom=0)
    
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

def plot_loss(loss_vals, ma_loss_vals, label, total_epochs, experiment_path,
              solver_title, switch_after=0, log_scale=False, rescaling=True, show_moving_averages=True):
    # Plots a single objective's values over the course of the training cycle
    if loss_vals is not None:
        loss_vals_count = len(loss_vals)
        if len(loss_vals) > 0:
            vals_per_epoch = round(loss_vals_count / total_epochs)

            if show_moving_averages and ma_loss_vals is not None:
                ma_loss_vals, ma_window_width = ma_loss_vals
            elif show_moving_averages:
                warnings.warn(f"Could not show moving averages because ma_loss_vals is {ma_loss_vals}")
            else: 
                ma_loss_vals, ma_window_width = None, None
            
            if rescaling:
                last_epoch_steps = int(len(loss_vals) / total_epochs)
                last_epoch_vals = loss_vals[-last_epoch_steps:]
                last_epoch_mean = sum(last_epoch_vals) / len(last_epoch_vals)
                divisor = last_epoch_mean if last_epoch_mean != 0 else 1
                loss_vals = np.divide(loss_vals, divisor) * 1000
                if show_moving_averages and ma_loss_vals is not None:
                    ma_loss_vals = np.divide(ma_loss_vals, divisor) * 1000

            loss_linewidth = 0.5 if show_moving_averages else 1.0
            plt.figure(figsize=(18, 8))
            plt.plot(loss_vals, label=label, linewidth=loss_linewidth, alpha=0.75)
            if show_moving_averages and ma_loss_vals is not None:
                plt.plot(ma_loss_vals, label=f"{label} (moving average, {ma_window_width})", linewidth=2)

            vline_width = 15 / total_epochs # 1.5 for 10 epochs
            for epoch in np.arange(0, total_epochs + 1):
                color = "r" if epoch == switch_after-1 else "black"
                plt.axvline(x=epoch*vals_per_epoch - 1, color=color, linewidth=1.5, linestyle="--")
            
            if log_scale:
                plt.yscale("log")
            else:
                plt.ylim(bottom=0)
            
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
                experiment_path, solver_title, epochs_before_switch=0, log_scale=False, show_moving_averages=True):
    # Plot losses
    print(f"Graphing overlaid losses...")
    switch_after = epochs_before_switch + 1

    # Plot all losses on one graph
    total_loss_vals = losses["Total Loss"]
    total_loss_ma = ma_losses["Total Loss"] if ma_losses is not None else None
    
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
    other_loss_ma = {key:ma_losses[key] for key in keys} if ma_losses is not None else None

    plot_overlaid_losses(total_loss_vals, total_loss_ma, other_loss_vals, other_loss_ma, total_epochs, 
                         experiment_path, solver_title, switch_after, log_scale, rescaling=False, 
                         show_moving_averages=show_moving_averages)

    # Plot individual losses
    print(f"Graphing total loss...")
    plot_loss(total_loss_vals, total_loss_ma, "Total Loss", total_epochs, experiment_path, 
              solver_title, switch_after, log_scale, rescaling=False, show_moving_averages=show_moving_averages)
    print(f"Graphing individual losses...")
    for key in keys:
        loss_vals = losses[key]
        loss_ma = ma_losses[key] if ma_losses is not None else None
        plot_loss(loss_vals, loss_ma, key, total_epochs, experiment_path, 
                  solver_title, switch_after, log_scale, rescaling=False, 
                  show_moving_averages=show_moving_averages)

    # Repeat for rescaled versions
    print(f"Graphing overlaid rescaled losses...")
    plot_overlaid_losses(total_loss_vals, total_loss_ma, other_loss_vals, other_loss_ma, total_epochs, 
                         experiment_path, solver_title, switch_after, log_scale, rescaling=True, 
                         show_moving_averages=show_moving_averages)
    print(f"Graphing rescaled total loss...")
    plot_loss(total_loss_vals, total_loss_ma, "Total Loss", total_epochs, experiment_path, 
              solver_title, switch_after, log_scale, rescaling=True, show_moving_averages=show_moving_averages)
    print(f"Graphing rescaled individual losses...")
    for key in keys:
        loss_vals = losses[key]
        loss_ma = ma_losses[key] if ma_losses is not None else None
        plot_loss(loss_vals, loss_ma, key, total_epochs, experiment_path, 
                  solver_title, switch_after, log_scale, rescaling=True, 
                  show_moving_averages=show_moving_averages)

