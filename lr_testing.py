import os
from ruamel.yaml import YAML
from bidcell import BIDCellModel

cwd = os.getcwd()
xenium_config_path = "xenium_example_config.yaml"

# Create temp yaml variants
temp_config_paths = []
learning_rates = [0.0001, 0.00001, 0.000001]

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
for temp_config_path in temp_config_paths:
	model = BIDCellModel(temp_config_path)
	model.preprocess()
	model.train()
	del model

# Delete temp yaml files
for temp_config_path in temp_config_paths:
	os.remove(temp_config_path)
