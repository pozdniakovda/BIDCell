import os
from ruamel.yaml import YAML
from bidcell import BIDCellModel

cwd = os.getcwd()
xenium_config_path = "xenium_example_config.yaml"

# Create temp yaml variants
temp_config_paths = []
loss_weight_keys = ["ne_weight", "os_weight", "cc_weight", "ov_weight", "pos_weight", "neg_weight"]

print(f"Creating temporary configs where a single loss is held at weight 0.5 while the remaining 5 losses are held at 0.1, with or without the Procrustes Solver")
for solver in ["default", "procrustes"]:
    for loss_weight_key in loss_weight_keys:
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(os.path.join(cwd, xenium_config_path), "r") as file:
        config = yaml.load(file)

    config["training_params"]["solver"] = solver
    for key in loss_weight_keys:
        config["training_params"][key] = 0.1
    config["training_params"][loss_weight_key] = 0.5

    temp_config_path = xenium_config_path.split(".yaml")[0] + f"_{solver}_emphasis={loss_weight_key}.yaml"
    with open(temp_config_path, "w") as file:
        yaml.dump(config, file)
    temp_config_paths.append(temp_config_path)

# Do BIDCell training for each variant
print(f"Starting BIDCell variant training...")
for temp_config_path in temp_config_paths:
    print(f"Training using config: {temp_config_path}")
    model = BIDCellModel(temp_config_path)
    print(f"\tPreprocessing data...")
    model.preprocess()
    print(f"\tTraining model...")
    model.train()
    print(f"\tDone this variant.")
    del model

# Delete temp yaml files
print(f"Cleaning up temp config files...")
for temp_config_path in temp_config_paths:
    os.remove(temp_config_path)
