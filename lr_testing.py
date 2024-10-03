import os
from ruamel.yaml import YAML
from bidcell import BIDCellModel

cwd = os.getcwd()
xenium_config_path = "xenium_example_config.yaml"

# Create temp yaml variants
temp_config_paths = []
learning_rates = [0.0001, 0.00001, 0.000001]

print(f"Creating temporary configs for lr=0.0001, lr=0.00001, and lr=0.000001, with or without the Procrustes Solver")
for solver in ["default", "procrustes"]:
	for learning_rate in learning_rates:
		yaml = YAML()
		yaml.preserve_quotes = True
		with open(os.path.join(cwd, xenium_config_path), "r") as file:
		    config = yaml.load(file)

		config["training_params"]["solver"] = solver
		config["training_params"]["learning_rate"] = learning_rate

		temp_config_path = xenium_config_path.split(".yaml")[0] + f"_{solver}_lr={learning_rate}.yaml"
		with open(temp_config_path, "w") as file:
		    yaml.dump(config, file)
		temp_config_paths.append(temp_config_path)

# Do BIDCell training for each variant
print(f"Starting BIDCell variant training...")
for temp_config_path in temp_config_paths:
	print(f"Training using config: {temp_config_path}")
	model = BIDCellModel(temp_config_path)
	print(f"Preprocessing data...")
	model.preprocess()
	print(f"Training model...")
	model.train()
	print(f"Done this variant.")
	del model

# Delete temp yaml files
print(f"Cleaning up temp config files...")
for temp_config_path in temp_config_paths:
	os.remove(temp_config_path)
