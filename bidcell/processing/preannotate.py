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


def normalise_matrix(matrix, replace_zeros=True):
    x_sums = np.sum(matrix, axis=1)
    x_sums_nan = np.isnan(x_sums)
    if x_sums_nan.all():
        print(f"Warning: entire `x_sums` is NaN.")
    elif x_sums_nan.any():
        print(f"Warning: `x_sums` contains {x_sums_nan.sum()} ({100*x_sums_nan.mean():.2f}%) NaN values. See `x_sums`:")
        print(x_sums)

    x_sums_zero = x_sums == 0
    if x_sums_zero.all():
        print(f"Warning: entire `x_sums` is zero (0).")
    elif x_sums_zero.any():
        print(f"Warning: `x_sums` contains {x_sums_zero.sum()} ({100*x_sums_zero.mean():.2f}%) zeros. See `x_sums`:")
        print(x_sums)
        
    if x_sums_zero.any() and replace_zeros:
        x_sums[x_sums == 0] = 1
        print(f"Replaced `x_sums` zeros (n={x_sums_zero.sum()}) with ones to prevent division-by-zero errors.")
    
    matrix = matrix / np.expand_dims(x_sums, -1)
    matrix_nan = np.isnan(matrix)
    if matrix_nan.all():
        print(f"Warning: after being divided by `x_sums`, entire `matrix` is NaN.")
    elif matrix_nan.any():
        print(f"Warning: after being divided by `x_sums`, `matrix` contains {matrix_nan.sum()} ({100*matrix_nan.mean():.2f}%) NaN values. See `matrix`:")
        print(matrix)
    
    matrix = np.log1p(matrix)
    matrix_nan = np.isnan(matrix)
    if matrix_nan.all():
        print(f"Warning: after applying np.log1p(), entire `matrix` is NaN.")
    elif matrix_nan.any():
        print(f"Warning: after applying np.log1p(), `matrix` contains {matrix_nan.sum()} ({100*matrix_nan.mean():.2f}%) NaN values. See `np.log1p(matrix)`:")
        print(matrix)
    
    return matrix


def process_chunk_corr(matrix, dir_output, sc_expr, sc_labels, n_atlas_types, save_chunk=False):
    matrix_out = np.zeros((matrix.shape[0], 4))
    col_names = ["cell_id", "cell_type", "spearman", "cell_type_atlas"]
    
    # Check for NaN in input data
    sc_expr_nan = np.isnan(sc_expr)
    if sc_expr_nan.all():
        print(f"Warning: entire `sc_expr` is NaN.")
    elif sc_expr_nan.any():
        print(f"Warning: `sc_expr` contains {sc_expr_nan.sum()} ({100*sc_expr_nan.mean():.2f}%) NaN values.")
        
    matrix_nan = np.isnan(matrix[:, 1:])
    if matrix_nan.all():
        print(f"Warning: entire `matrix` is NaN.")
    elif matrix_nan.any():
        print(f"Warning: `matrix` contains {matrix_nan.sum()} ({100*matrix_nan.mean():.2f}%) NaN values.")

    # cell_type
    cell_genes_norm = normalise_matrix(matrix[:, 1:])
    cg_norm_nan = np.isnan(cell_genes_norm)
    if cg_norm_nan.all():
        print(f"Warning: entire `cell_genes_norm` is NaN.")
    elif cg_norm_nan.any():
        print(f"Warning: `cell_genes_norm` contains {cg_norm_nan.sum()} ({100*cg_norm_nan.mean():.2f}%) NaN values.")

    # Check for zero variance rows (which can cause NaN in Spearman)
    zero_var_sc_expr = np.where(np.std(sc_expr, axis=1) == 0)[0]
    zero_var_matrix = np.where(np.std(cell_genes_norm, axis=1) == 0)[0]
    if len(zero_var_sc_expr) > 0:
        print(f"Warning: {len(zero_var_sc_expr)} rows (of {len(sc_expr)}) in `sc_expr` have zero variance.")
    if len(zero_var_matrix) > 0:
        print(f"Warning: {len(zero_var_matrix)} rows (of {len(cell_genes_norm)}) in `cell_genes_norm` have zero variance.")
    
    res = spearmanr(sc_expr, cell_genes_norm, axis=1)
    corr = res.correlation

    corr_nan = np.isnan(corr)
    if corr_nan.all():
        print(f"Warning: Spearman correlation matrix is entirely NaN.")
    elif corr_nan.any():
        print(f"Warning: Spearman correlation matrix contains {corr_nan.sum()} ({100*corr_nan.mean():.2f}%) NaN values.")
    
    # bottom left section
    corr = corr[n_atlas_types:, :n_atlas_types]

    corr_best = np.max(corr, 1)
    best_i_type = np.argmax(corr, 1)
    predicted_cell_type = [sc_labels[x] for x in best_i_type]

    nan_true = np.isnan(corr_best)
    num_nan = nan_true.sum()
    if num_nan > 0:
        print(f"Warning: {num_nan} ({100*nan_true.mean():.2f}%) cells have NaN correlations and will be assigned -1.")

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
    df_split = pd.DataFrame(
        matrix_out, index=list(range(matrix_out.shape[0])), columns=col_names
    )
    if save_chunk:
        fp_anno = dir_output + "/preannotations_%d.csv" % matrix_out[0, 0]
        df_split.to_csv(fp_anno, index=False)
        #print(f"Saved preannotations file: {fp_anno}")
    else: 
        fp_anno = None

    return (df_split, fp_anno)


def preannotate(config: Config, is_cell: bool = False, timestamp: str | None = None, 
                save_merged = False, save_chunks = False):
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
    remove_cols = ["cell_centroid_x", "cell_centroid_y", "cell_size", "pixel_size"]
    for col in remove_cols:
        if col in genes_cells:
            genes_cells.remove(col)
    
    ct_columns = df_ref_orig.columns[-3:].tolist()
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

    print(f"genes_cells len={len(genes_cells)}")

    sc_expr = df_ref.iloc[:, :-3].to_numpy()
    n_atlas_types = sc_expr.shape[0]
    sc_labels = df_ref.iloc[:, -3].to_numpy().astype(int)
    # sc_names = df_ref.iloc[:, -2].to_list()

    # Divide the data into chunks for multiprocessing
    n_processes = get_n_processes(config.cpus)
    print(f"Number of splits for multiprocessing: {n_processes}")

    matrix_all = df_cells[["cell_id"] + genes_cells].to_numpy().astype(np.float32)
    matrix_all_splits = np.array_split(matrix_all, n_processes)

    print("Computing simple annotation")
    args_list = [(chunk, dir_dataset, sc_expr, sc_labels, n_atlas_types, save_chunks) for chunk in matrix_all_splits]
    cell_dfs = []
    anno_fps = []
        
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for df_split, fp_anno in pool.starmap(process_chunk_corr, args_list):
            cell_dfs.append(df_split)
            anno_fps.append(fp_anno)
    
    cell_df = pd.concat(cell_dfs, ignore_index=True)
    print(f"Annotated cell_df:")
    print(cell_df)

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
        if is_cell:
            cell_df.to_csv(dir_dataset + "/preannotations_cell_merged.csv")
        else:
            cell_df.to_csv(dir_dataset + "/preannotations_nuclei_merged.csv")

    return cell_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")

    args = parser.parse_args()
    config = load_config(args.config_dir)

    preannotate(config)
