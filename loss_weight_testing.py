import numpy as np
import os
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
from bidcell import BIDCellModel

cwd = os.getcwd()
xenium_config_path = "xenium_example_config.yaml"

yaml = YAML()
yaml.preserve_quotes = True
with open(os.path.join(cwd, xenium_config_path), "r") as file:
    config = yaml.load(file)
epochs = config["training_params"]["total_epochs"]
solver = config["training_params"]["solver"]

# Create temp yaml variants
temp_config_paths = []

loss_weights_arrs = np.full(shape=(6, 6), fill_value=0.1, dtype=float)
np.fill_diagonal(loss_weights_arrs, 0.5)
equal_weights_arr = np.full(shape=6, fill_value=(1/6), dtype=float)
loss_weights_arrs = np.vstack([equal_weights_arr, loss_weights_arrs])

loss_weight_keys = ["ne_weight", "os_weight", "cc_weight", "ov_weight", "pos_weight", "neg_weight"]
emphasized_keys = ["equal"] + loss_weight_keys
emphasized_colors = ["black", "blue", "magenta", "darkgreen", "orange", "green", "red"] # equal, ne_weight, os_weight, cc_weight, ov_weight, pos_weight, neg_weight

print(f"Creating temporary configs where a single loss is held at weight 0.5 while the remaining 5 losses are held at 0.1")
for emphasized_key, color, loss_weights in zip(emphasized_keys, emphasized_colors, loss_weights_arrs):
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(os.path.join(cwd, xenium_config_path), "r") as file:
        config = yaml.load(file)

    loss_weights = list(loss_weights)
    for key, weight in zip(loss_weight_keys, loss_weights):
        config["training_params"][key] = float(weight)

    temp_config_path = xenium_config_path.split(".yaml")[0] + f"_{solver}_emphasis={emphasized_key}.yaml"
    with open(temp_config_path, "w") as file:
        yaml.dump(config, file)
    temp_config_paths.append((temp_config_path, emphasized_key, color))

# Do BIDCell training for each variant
print(f"Starting BIDCell variant training...")
graphing_data = []
for temp_config_path, emphasized_key, color in temp_config_paths:
    print(f"Training using config: {temp_config_path}")
    model = BIDCellModel(temp_config_path)
    
    print(f"\tPreprocessing data...")
    model.preprocess()
    
    print(f"\tTraining model...")
    model.train()
    graphing_data.append((emphasized_key, color, model.loss_histories, model.ma_loss_histories, model.experiment_path))
    
    print(f"\tDone this variant.")
    del model

# Delete temp yaml files
print(f"Cleaning up temp config files...")
for temp_config_path in temp_config_paths:
    os.remove(temp_config_path[0])

# Generate the overlaid graphs
print(f"Generating overlaid graphs...")
graph_data_keys = ["Total Loss", "Nuclei Encapsulation Loss", "Oversegmentation Loss", "Cell Calling Loss", "Overlap Loss", "Pos-Neg Marker Loss"]

for graph_data_key in graph_data_keys: 
    plt.figure(figsize=(18, 8))
    for emphasized_key, color, loss_histories, _, experiment_path in graphing_data: 
        parent_path = os.path.dirname(experiment_path)
        steps = len(loss_histories[graph_data_key])
        
        label = f"{graph_data_key}, emphasis = {emphasized_key}"
        
        histories = loss_histories[graph_data_key]
        histories = np.array(histories)
        scaled_histories = histories / histories.max() if histories.max() != 0 else histories
        
        plt.plot(scaled_histories, label=label, linewidth=0.5, color=color, alpha=0.5)

    for emphasized_key, color, _, ma_loss_histories, experiment_path in graphing_data: 
        ma_loss_vals, ma_window_width = ma_loss_histories[graph_data_key]
        ma_loss_vals = np.array(ma_loss_vals)
        ma_label = f"{graph_data_key}, emphasis = {emphasized_key}, (moving average, {ma_window_width})"

        scaled_moving_avgs = ma_loss_vals / ma_loss_values.max() if ma_loss_values.max() != 0 else ma_loss_vals
        plt.plot(ma_loss_vals, label=ma_label, linewidth=2, color=color)

    steps_per_epoch = round(epochs / steps)
    for epoch in range(epochs):
        plt.axvline(x=epoch*steps_per_epoch, color="darkgray", linestyle="--", alpha=0.5)
    
    plt.xlabel("Training Step")
    plt.ylabel("Normalized Loss")
    
    if solver == "procrustes":
        plt.title(f"{graph_data_key} During Training with Procrustes Method (Emphasis: {emphasized_key})")
    else: 
        plt.title(f"{graph_data_key} During Training with Default Method (Emphasis: {emphasized_key})")

    plt.yscale("log")
    
    plt.tight_layout()        
    filename = "training_" + "_".join(graph_data_key.lower().split(" ")) + "_overlaid.pdf"
    plt.savefig(os.path.join(parent_path, filename))
