"""BIDCellModel class module"""
import importlib.resources
import os
import pandas as pd
from pathlib import Path
from shutil import copyfile, copytree
from typing import Literal

from .config import load_config
from .model.postprocess_predictions import postprocess_predictions
from .model.predict import fill_grid, predict
from .model.train import train
from .model.utils.utils import get_newest_id
from .processing.cell_gene_matrix import make_cell_gene_mat
from .processing.nuclei_segmentation import segment_nuclei
from .processing.nuclei_stitch_fov import stitch_nuclei
from .processing.preannotate import preannotate
from .processing.transcript_patches import generate_patches
from .processing.transcripts import generate_expression_maps


class BIDCellModel:
    """The BIDCellModel class, which provides an interface for preprocessing, training and predicting all the cell types for a datset."""

    def __init__(self, config_file = None, config_files = None, lr_override = None, solver_override = None, 
                 device_idx = None, verbose = False) -> None:
        """Constructs a BIDCellModel instance using the user-supplied config file.\n
        The configuration is validated during construction.

        Parameters
        ----------
        config_file : str
            Path to the YAML configuration file.
        """
        self.config = load_config(config_file) if config_file is not None else None
        self.config_files = config_files
        self.config_history = [config_file]
        self.lr_override = lr_override
        self.solver_override = solver_override
        self.device_idx = None
        self.verbose = False

    def replace_config(self, config_file: str, remove_overrides=True):
        """Replaces the config with a new one; useful for retraining loops. 
        """
        self.config = load_config(config_file)
        self.config_history.append(config_file)
        if remove_overrides:
            self.lr_override = None
            self.solver_override = None

    def run_pipeline(self):
        """Runs the entire BIDCell pipeline using the settings defined in the configuration.
        """
        print("### Preprocessing ###")
        print()
        self.preprocess()
        print()
        print("### Training ###")
        print()
        self.train()
        print()
        print("### Predict ###")
        print()
        self.predict()
        print()
        print("### Done ###")
    
    def rerun_pipeline(self):
        """Re-runs the entire BIDCell pipeline (except preprocessing) using the settings defined in the configuration.
        """
        print("### Training ###")
        print()
        self.train()
        print()
        print("### Predict ###")
        print()
        self.predict()
        print()
        print("### Done ###")
        
    def run_multiple(self, config_files = None):
        """Runs the entire BIDCell pipeline using the settings defined in the configuration.
        """
        if config_files is None and self.config_files is not None:
            config_files = self.config_files
        elif config_files is None:
            raise Exception(f"model.run_multiple() requires self.config_files to exist or for "
                            "config_files to be passed as an argument, but both were None.")
        else:
            self.config_files = config_files
        
        print("### Preprocessing ###")
        print()
        self.replace_config(config_files[0])
        self.preprocess()
        print()
        
        for config_file in config_files:
            print("### Training config: {config_file} ###")
            self.replace_config(config_file)
            self.train()
            print()
            
        # print("### Predict ###")
        # print()
        # self.predict()
        # print()
        print("### Done ###")

    def preprocess(self) -> None:
        """Preprocess the dataset for training.
        """
        if self.config.nuclei_fovs.stitch_nuclei_fovs:
            stitch_nuclei(self.config)
        if self.config.nuclei.crop_nuclei_to_ts:
            generate_expression_maps(self.config)
            segment_nuclei(self.config)
        else:
            segment_nuclei(self.config)
            generate_expression_maps(self.config)
        generate_patches(self.config)
        make_cell_gene_mat(self.config, is_cell=False)
        preannotate(self.config, save_merged=True)

    def stitch_nuclei(self):
        """Stich separate FOV files into a single one (e.g. CosMx data).\n
        Runs inside preprocess by default, if nuclei_fovs.stitch_nuclei_fovs is True in the configuration file.
        """
        stitch_nuclei(self.config)

    def segment_nuclei(self):
        """Run the nucleus segmentation algorythm. Runs inside preprocess by default.
        """
        segment_nuclei(self.config)

    def generate_expression_maps(self):
        """Generate the expression maps. Runs inside preprocess by default.
        """
        generate_expression_maps(self.config)

    def generate_patches(self):
        """Generate patches for training. Runs inside preprocess by default.
        """
        generate_patches(self.config)

    def make_cell_gene_mat(self, is_cell: bool, timestamp: str = "last"):
        """Make a matrix containing counts for each cell. Runs inside preprocess and predict by default.

        Parameters
        ----------
        is_cell : bool
            If False, uses nuclei masks for creation, other wise it uses `timestamp` to chose a directory containing segmented cells outputted by BIDCell.
        timestamp : str, optional
            The timestamp corrisponding to the name of a directory in the data directory under `model_outputs`, by default "last", in which case it uses the folder with the most recent timestamp.
        """
        if is_cell and timestamp == "last":
            timestamp = get_newest_id(
                os.path.join(self.config.files.data_dir, "model_outputs")
            )
        elif is_cell:
            self.__check_valid_timestamp(timestamp)
        make_cell_gene_mat(self.config, is_cell, timestamp=timestamp)

    def preannotate(self):
        """Preannotate the cells. Runs inside preprocess by default.
        """
        preannotate(self.config)

    def train(self) -> None:
        """Train the model.
        """
        self.loss_histories, self.ma_loss_histories, self.experiment_path = train(self.config, 
                                                                                  self.lr_override, 
                                                                                  self.solver_override, 
                                                                                  self.device_idx, self.verbose)

    def predict(self) -> None:
        """Segment and annotate the cells, then merge all metadata into expr_mat.csv."""
        print(f"Beginning prediction...")
        predict(self.config)
    
        if self.config.experiment_dirs.dir_id == "last":
            timestamp = get_newest_id(
                os.path.join(self.config.files.data_dir, "model_outputs")
            )
        else:
            timestamp = self.config.experiment_dirs.dir_id
            self.__check_valid_timestamp(timestamp)
    
        print(f"Filling grid...")
        fill_grid(self.config, timestamp)
    
        print(f"Postprocessing predictions...")
        postprocess_predictions(self.config, timestamp)

        print(f"Making cell gene matrix...")
        make_cell_gene_mat(self.config, is_cell=True, timestamp=timestamp)
        
        print(f"Re-running preannotation...")
        anno_df = preannotate(self.config, is_cell=True, timestamp=timestamp, save_merged=True)

        print(f"Reloading cell gene matrix to include annotations...")
        expr_mat_path = os.path.join(self.config.files.data_dir, self.config.files.dir_cgm, timestamp, self.config.files.fp_expr)
        preannotations_path = os.path.join(self.config.files.data_dir, "preannotations_merged.csv")
    
        if os.path.exists(expr_mat_path) and os.path.exists(preannotations_path):
            df_expr = pd.read_csv(expr_mat_path)
            df_annotations = pd.read_csv(preannotations_path)
    
            # Merge on cell_id
            df_merged = df_expr.merge(df_annotations, on="cell_id", how="left")
    
            # Save the updated expr_mat.csv
            df_merged.to_csv(expr_mat_path, index=False)
            print(f"Merged annotations into expr_mat.csv successfully! Save path: {expr_mat_path}")
        elif os.path.exists(expr_mat_path) and not os.path.exists(preannotations_path):
            print("Warning: preannotations_merged.csv not found, skipping merge.")
        elif not os.path.exists(expr_mat_path) and os.path.exists(preannotations_path):
            print("Warning: expr_mat.csv not found, skipping merge.")
        else:
            print("Warning: expr_mat.csv and preannotations_merged.csv not found, skipping merge.")
    
        print(f"Done prediction.")

    @staticmethod
    def get_example_config(vendor: Literal["cosmx", "merscope", "stereoseq", "xenium"]) -> None:
        """Gets an example configuration for a given vendor and places it in the working directory.

        Parameters
        ----------
        vendor : Literal["cosmx", "merscope", "stereoseq", "xenium"]
            The vendor of the equiptment used to produce the dataset.
        """
        vendors = ["cosmx", "merscope", "stereoseq", "xenium"]
        if not any([vendor.lower() == x for x in vendors]):
            raise ValueError(f"Unknown vendor `{vendor}`\n\tChose one of {*vendors,}")
        params_path = (
            importlib.resources.files("bidcell") / "example_params" / f"{vendor}.yaml"
        )
        if not (dest := Path().cwd() / f"{vendor}_example_config.yaml").exists():
            copyfile(params_path, dest)

    @staticmethod
    def get_example_data(with_config: bool = True) -> None:
        """Gets the small example data included in the package and places it in the current working directory.

        Parameters
        ----------
        with_config : bool, optional
            Whether to get the configuration for the example data, by default True
        """
        root: Path = importlib.resources.files("bidcell")
        data_path = (
            root.parent / "data"
        )
        cwd = Path().cwd()
        if not (cwd / "example_data").exists():
            copytree(data_path, cwd / "example_data")
        if with_config and not (cwd / "params_small_example.yaml").exists():
            copyfile(
                root / "example_params" / "small_example.yaml",
                cwd / "params_small_example.yaml"
            )

    def __check_valid_timestamp(self, timestamp: str) -> None:
        outputs_path = Path(self.config.files.data_dir / "model_outputs")
        outputs = list(outputs_path.iterdir())
        if len(outputs) == 0:
            raise ValueError(
                f"There are no outputs yet under {str(outputs_path)}. Run BIDCell at least once with this dataset to get some."
            )
        if not any(
            [timestamp == x for x in outputs if x.is_dir()]
        ):
            valid_dirs = "\n".join(["\t" + str(x) for x in outputs])
            raise ValueError(
                f"{timestamp} is not a valid model output directory (set in configuration YAML under `experiment_dirs.dir_id`). Choose one of the following:\n{valid_dirs}"
            )
