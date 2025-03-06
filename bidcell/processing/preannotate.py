import argparse
import collections
import glob
import json
import multiprocessing as mp
import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .utils import get_n_processes
from ..config import Config, load_config

np.seterr(divide="ignore", invalid="ignore")


def json_file_to_pyobj(filename):
    """
    Read json config file
    """

    def _json_object_hook(d):
        return collections.namedtuple("X", d.keys())(*d.values())

    def json2obj(data):
        return json.loads(data, object_hook=_json_object_hook)

    return json2obj(open(filename).read())


def normalise_matrix(matrix):
    x_sums = np.sum(matrix, axis=1)
    matrix = matrix / np.expand_dims(x_sums, -1)
    matrix = np.log1p(matrix)
    return matrix


def process_chunk_corr(matrix, dir_output, sc_expr, sc_labels, n_atlas_types):
    matrix_out = np.zeros((matrix.shape[0], 4))
    col_names = ["cell_id", "cell_type", "spearman", "cell_type_atlas"]

    # cell_type
    cell_genes_norm = normalise_matrix(matrix[:, 1:])
    res = spearmanr(sc_expr, cell_genes_norm, axis=1)
    corr = res.correlation
    # bottom left section
    corr = corr[n_atlas_types:, :n_atlas_types]

    corr_best = np.max(corr, 1)
    best_i_type = np.argmax(corr, 1)
    predicted_cell_type = [sc_labels[x] for x in best_i_type]

    nan_true = np.isnan(corr_best)
    corr_best = [x if not y else -1 for (x, y) in zip(corr_best, nan_true)]
    best_i_type = [x if not y else -1 for (x, y) in zip(best_i_type, nan_true)]
    predicted_cell_type = [
        x if not y else -1 for (x, y) in zip(predicted_cell_type, nan_true)
    ]

    # cell ID
    matrix_out[:, 0] = matrix[:, 0].copy()

    # cell type
    matrix_out[:, 1] = predicted_cell_type.copy()

    # spearman
    matrix_out[:, 2] = corr_best.copy()

    # cell type atlas
    matrix_out[:, 3] = best_i_type.copy()

    # Save as csv
    fp_anno = dir_output + "/preannotations_%d.csv" % matrix_out[0, 0]
    df_split = pd.DataFrame(
        matrix_out, index=list(range(matrix_out.shape[0])), columns=col_names
    )
    df_split.to_csv(fp_anno, index=False)
    #print(f"Saved preannotations file: {fp_anno}")


def preannotate(config: Config, is_cell: bool = False, timestamp: str | None = None, save_merged = False):
    dir_dataset = config.files.data_dir
    dir_cgm = config.files.dir_cgm
    
    if is_cell is False:
        expr_dir = os.path.join(dir_dataset, dir_cgm, "nuclei")
    else:
        expr_dir = os.path.join(dir_dataset, dir_cgm, timestamp)

    # Cell expressions - order of gene names (columns) will be in same order as all_gene_names.txt
    fp_expr = os.path.join(expr_dir, config.files.fp_expr)
    print(f"Loading cell expressions from fp_expr: {fp_expr}")
    df_cells = pd.read_csv(fp_expr, index_col=0)
    print(f"Number of cells: {df_cells.shape[0]}")

    # Reference data - no requirement of column orders - ensure same order as df_cells
    fp_ref = config.files.fp_ref
    print(f"Loading reference from fp_expr: {fp_ref}")
    df_ref_orig = pd.read_csv(fp_ref, index_col=0)

    # Ensure the order of genes match
    genes_cells = df_cells.columns[1:].tolist()
    print("genes_cells:")
    print(genes_cells)
    
    ct_columns = df_ref_orig.columns[-3:].tolist()
    print("ct_columns:")
    print(ct_columns)
    
    df_ref = df_ref_orig[genes_cells + ct_columns]

    genes_ref = df_ref.columns[:-3]
    if list(genes_cells) != list(genes_ref):
        print(
            "Genes in transcripts but not reference: ",
            list(set(genes_cells) - set(genes_ref)),
        )
        print(
            "Genes in reference but not transcripts: ",
            list(set(genes_ref) - set(genes_cells)),
        )
        print("Check names of genes")
        sys.exit()

    sc_expr = df_ref.iloc[:, :-3].to_numpy()
    n_atlas_types = sc_expr.shape[0]
    sc_labels = df_ref.iloc[:, -3].to_numpy().astype(int)
    # sc_names = df_ref.iloc[:, -2].to_list()

    # Divide the data into chunks for multiprocessing
    n_processes = get_n_processes(config.cpus)
    print(f"Number of splits for multiprocessing: {n_processes}")

    matrix_all = df_cells.to_numpy().astype(np.float32)
    matrix_all_splits = np.array_split(matrix_all, n_processes)
    processes = []

    print("Computing simple annotation")
    for chunk in matrix_all_splits:
        p = mp.Process(
            target=process_chunk_corr,
            args=(chunk, dir_dataset, sc_expr, sc_labels, n_atlas_types),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    fp_chunks = glob.glob(dir_dataset + "/preannotations_*.csv")
    for fp_i, fpc in enumerate(fp_chunks):
        df_i = pd.read_csv(fpc)
        if fp_i == 0:
            cell_df = df_i.copy()
        else:
            cell_df = pd.concat([cell_df, df_i], axis=0)

    cell_type_col = cell_df["cell_type"].to_numpy()
    cell_id_col = cell_df["cell_id"].to_numpy()

    fp_anno = config.files.fp_cell_anno if is_cell else config.files.fp_nuclei_anno
    h5f_path = dir_dataset + "/" + fp_anno
    
    h5f = h5py.File(h5f_path, "w")
    h5f.create_dataset("data", data=cell_type_col)
    h5f.create_dataset("ids", data=cell_id_col)
    h5f.close()

    # Save merged dataframe
    if save_merged:
        preannotation_dfs = [pd.read_csv(file_path) for file_path in fp_chunks]
        preannotation_df = pd.concat(preannotation_dfs, ignore_index=True)
        preannotation_df.to_csv(dir_dataset + "/preannotations_merged.csv")
    
    # Clean up
    for fpc in fp_chunks:
        os.remove(fpc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")

    args = parser.parse_args()
    config = load_config(args.config_dir)

    preannotate(config)
