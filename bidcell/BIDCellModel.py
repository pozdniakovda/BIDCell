"""BIDCellModel class module"""
from typing import Optional
from multiprocessing import cpu_count
import os

import yaml

from .model.postprocess_predictions import postprocess_predictions
from .model.predict import predict
from .model.train import train
from .processing.cell_gene_matrix import make_cell_gene_mat
from .processing.nuclei_segmentation import segment_nuclei
from .processing.nuclei_stitch_fov import stitch_nuclei
from .processing.preannotate import preannotate
from .processing.transcript_patches import generate_patches
from .processing.transcripts import generate_expression_maps
from .model.utils.utils import get_newest_id
from .config import load_config


class BIDCellModel:
    """The BIDCellModel class, which provides an interface for preprocessing, training and predicting all the cell types for a datset."""

    def __init__(self, config_file: str, n_processes: Optional[int] = None) -> None:
        self.config = load_config(config_file)

        if n_processes is None:
            self.n_processes = cpu_count()
        else:
            self.n_processes = n_processes

    def preprocess(self) -> None:
        if self.config.nuclei_fovs.stitch_nuclei_fovs:
            stitch_nuclei(self.config)
        segment_nuclei(self.config)
        generate_expression_maps(self.config)
        generate_patches(self.config)
        make_cell_gene_mat(
            self.config, is_cell=True
        )  # TODO: set config.cgm_params.only_expr (True for nuclei)
        preannotate(self.config)
        # TODO: Which information do the end users need from the process?

    def stitch_nuclei(self):
        stitch_nuclei(self.config)

    def segment_nuclei(self):
        segment_nuclei(self.config)

    def generate_expression_maps(self):
        generate_expression_maps(self.config)

    def generate_patches(self):
        generate_patches(self.config)

    def make_cell_gene_mat(self, is_cell=True):
        make_cell_gene_mat(self.config, is_cell)

    def preannotate():
        preannotate(self.config)

    def train(self) -> None:
        train(self.config)

    def predict(self) -> None:
        predict(self.config)
        # TODO: figure out the most recent experiment. get_lastest_id()
        if self.config.postprocess.dir_id == "last":
            self.config.postprocess.dir_id = get_newest_id()
        postprocess_predictions(self.config)
        # TODO: Figure out final cell_gene_matrix call
        # config.files.dir_output_matrices
        # config.files.fp_seg
        # config.cgm_params.only_expr (False for cells)

    def set_config() -> None:
        # TODO: Document all config options and allow setting single or
        #       multiple options at a time.
        raise NotImplementedError()

    def __repr__(self) -> str:
        """Returns formatted BIDCellModel as a string with
        key configuration options listed as well as information about completed
        steps.

        Returns
        -------
        str"""
        return "Not implemented yet!"
