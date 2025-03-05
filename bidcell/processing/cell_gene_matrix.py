import argparse
import glob
import os
import sys
import time
import multiprocessing as mp
mp.set_start_method("forkserver", force=True)

# import cv2
from skimage.measure import regionprops, label
from skimage.transform import resize
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from .utils import get_n_processes, get_patches_coords
from ..config import Config, load_config

np.seterr(divide="ignore", invalid="ignore")


def prepare_expr(seg_map_full, hs, he, ws, we, fp_transcripts_processed, x_col, y_col, 
                 scale_pix_x, scale_pix_y, print_ranges=False):
    # Prepares segmentation submap and expressions for processing
    
    print(f"Patch H {hs}:{he}, W {ws}:{we}")
    seg_map = seg_map_full[hs:he, ws:we]
    print(seg_map.shape)

    df_expr = read_expr_csv(fp_transcripts_processed)
    if print_ranges:
        print(
            df_expr[x_col].min(),
            df_expr[x_col].max(),
            df_expr[y_col].min(),
            df_expr[y_col].max(),
        )

    df_expr = transform_locations(df_expr, x_col, scale_pix_x)
    df_expr = transform_locations(df_expr, y_col, scale_pix_y)

    df_expr = df_expr[
        (df_expr[x_col].between(ws, we - 1))
        & (df_expr[y_col].between(hs, he - 1))
    ]
    if print_ranges:
        print(
            df_expr[x_col].min(),
            df_expr[x_col].max(),
            df_expr[y_col].min(),
            df_expr[y_col].max(),
        )

    df_expr = transform_locations(df_expr, x_col, 1, ws)
    df_expr = transform_locations(df_expr, y_col, 1, hs)
    if print_ranges: 
        print(
            df_expr[x_col].min(),
            df_expr[x_col].max(),
            df_expr[y_col].min(),
            df_expr[y_col].max(),
        )

    return seg_map, df_expr


def process_chunk(chunk, output_dir, cell_ids_unique, col_names, seg_map, 
                  x_col, y_col, gene_col, save_chunk=False):
    """Extract cell expression profiles"""
    
    df_out = pd.DataFrame(0, index=cell_ids_unique, columns=col_names)
    df_out["cell_id"] = cell_ids_unique.copy()

    chunk_id = chunk.index[0]

    for _, row in chunk.iterrows():
        gene = row[gene_col]
        w_loc = row[x_col]
        h_loc = row[y_col]

        seg_val = seg_map[h_loc, w_loc]
        if seg_val > 0:
            df_out.loc[seg_val, gene] += 1

    # Check if any gene columns are completely empty
    if df_out.iloc[:, 1:].isnull().all().all():
        warning_message = f"All gene expression values are empty for chunk {chunk_id}!"
    else:
        warning_message = None

    if save_chunk:
        file_path = output_dir + f"/chunk_{chunk_id}.csv"
        df_out.to_csv(file_path)
    else:
        file_path = None

    return (df_out, file_path, warning_message)


def process_parallel(df_expr, n_processes, output_dir, cell_ids_unique, col_names, 
                     seg_map, x_col, y_col, gene_col):
    # Original parallelized data processing method
    df_expr_splits = np.array_split(df_expr, n_processes)
    processes = []

    print("Extracting cell-gene matrix chunks")
    for chunk in df_expr_splits:
        p = mp.Process(
            target=process_chunk,
            args=(
                chunk,
                output_dir,
                cell_ids_unique,
                col_names,
                seg_map,
                x_col,
                y_col,
                gene_col,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def process_starmap(df_expr, n_processes, output_dir, cell_ids_unique, col_names, 
                    seg_map, x_col, y_col, gene_col, save_chunks=False):
    """Parallelized CGM data processing using `starmap`, collecting and merging results."""

    df_expr_splits = np.array_split(df_expr, n_processes)
    results = []

    print("Extracting cell-gene matrix chunks")

    with mp.Pool(n_processes) as pool:
        args = [(chunk, output_dir, cell_ids_unique, col_names, seg_map, x_col, y_col, gene_col, save_chunks) for chunk in df_expr_splits]

        for df_out, file_path, warning_message in pool.starmap(process_chunk, args):
            if warning_message is not None:
                print(f"Warning: {warning_message}")

            if file_path:
                print(f"Processed and saved: {file_path}")
            # else:
            #     print(f"Processed results for chunk #{i+1}")
            
            results.append(df_out)

    # Merge results and keep only valid gene columns
    df_merged = pd.concat(results, ignore_index=True)
    df_merged = df_merged.loc[:, df_merged.notna().any()]  # Remove empty columns

    df_merged.set_index("cell_id", drop=True)

    return df_merged


def process_chunk_meta(matrix, fp_output, seg_map_mi, col_names_coords, 
                       scale_pix_x, scale_pix_y, cell_annotations=None, save_chunk=False):
    """Compute cell locations, sizes, eccentricity, and retrieve cell type annotations."""

    chunk_id = matrix[0, 0]
    
    # Convert to Pandas DataFrame (only metadata columns)
    df_output = pd.DataFrame(columns=col_names_coords)
    df_output["cell_id"] = matrix[:, 0].astype(int)

    for cur_i, cell_id in enumerate(df_output["cell_id"]):
        if cell_id > 0:
            try:
                # Extract spatial metadata
                coords = np.where(seg_map_mi == cell_id)
                df_output.at[cur_i, "cell_centroid_x"] = np.mean(coords[1]) * scale_pix_x
                df_output.at[cur_i, "cell_centroid_y"] = np.mean(coords[0]) * scale_pix_y
                df_output.at[cur_i, "pixel_size"] = len(coords[0]) / (scale_pix_x * scale_pix_y)
                df_output.at[cur_i, "eccentricity"] = regionprops(label(seg_map_mi == cell_id))[0].eccentricity
            except Exception:
                df_output.at[cur_i, ["cell_centroid_x", "cell_centroid_y", "pixel_size", "eccentricity"]] = [-1, -1, -1, -1]

    # Save only if needed
    if save_chunk:
        file_path = f"{fp_output}{chunk_id}.csv"
        df_output.to_csv(file_path, index=False)
    else:
        file_path = None

    return df_output, file_path


def process_parallel_meta(df_out, gene_names, n_processes, output_dir, seg_map_mi, 
                          scale_pix_x, scale_pix_y, cell_annotations=None):
    # Original meta processing where cell shape is measured and quantified
    
    matrix_all = df_out.to_numpy().astype(np.float32)
    matrix_all_splits = np.array_split(matrix_all, n_processes)
    processes = []

    fp_output = output_dir + "/cell_outputs_"
    col_names_coords = [
        "cell_id",
        "cell_centroid_x",
        "cell_centroid_y",
        "pixel_size",
        "eccentricity",
    ] + gene_names

    for chunk in matrix_all_splits:
        p = mp.Process(
            target=process_chunk_meta,
            args=(
                chunk,
                fp_output,
                seg_map_mi,
                col_names_coords,
                scale_pix_x,
                scale_pix_y,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    return col_names_coords


def process_starmap_meta(df_out, gene_names, n_processes, output_dir, seg_map_mi, 
                         scale_pix_x, scale_pix_y, cell_annotations=None, save_chunks=False):
    # Meta processing where cell shape is quantified; parallized with Starmap

    matrix_all = df_out.to_numpy().astype(np.float32)
    matrix_all_splits = np.array_split(matrix_all, n_processes)
    processes = []

    fp_output = output_dir + "/cell_outputs_"
    col_names_coords = [
        "cell_id",
        "cell_centroid_x",
        "cell_centroid_y",
        "pixel_size",
        "eccentricity",
    ] + gene_names

    results = []
    
    with mp.Pool(n_processes) as pool:
        args = []
        for chunk in matrix_all_splits:
            args.append((chunk, output_dir, seg_map_mi, col_names_coords, scale_pix_x, scale_pix_y, 
                         cell_annotations, save_chunks))

        for i, (df_out, file_path) in enumerate(pool.starmap(process_chunk_meta, args)):
            if file_path is not None:
                print(f"Processed and saved meta results for chunk #{i+1}: {file_path}")
            # else:
                # print(f"Processed meta results for chunk #{i+1}")
            
            results.append(df_out)

    # Concatenate all DataFrames
    df_merged = pd.concat(results, ignore_index=True)
    df_merged.set_index("cell_id", drop=True)

    return df_merged


def transform_locations(df_expr, col, scale, shift=0):
    """Scale transcripts to pixel resolution of the platform"""
    print(f"Transforming {col}")
    df_expr[col] = df_expr[col].div(scale).round().astype(int).sub(shift)
    return df_expr


def read_expr_csv(fp):
    try:
        print("Reading filtered transcripts")
        return pd.read_csv(fp)
    except Exception:
        sys.exit(f"Cannot read {fp}")


def get_cgm_paths(config: Config, is_cell: bool, timestamp: str | None = None):    
    dir_dataset = config.files.data_dir
    dir_cgm = config.files.dir_cgm

    if is_cell is False:
        output_dir = os.path.join(dir_dataset, dir_cgm, "nuclei")
    else:
        output_dir = os.path.join(dir_dataset, dir_cgm, timestamp)

    fp_transcripts_processed = os.path.join(
        dir_dataset, config.files.fp_transcripts_processed
    )

    fp_gene_names = os.path.join(dir_dataset, config.files.fp_gene_names)

    if is_cell is False:
        fp_seg = os.path.join(dir_dataset, config.files.fp_nuclei)
        fp_seg_name = None
    else:
        fp_seg_name = [
            "epoch_"
            + str(config.testing_params.test_epoch)
            + "_step_"
            + str(config.testing_params.test_step)
            + "_connected.tif"
        ]
        fp_seg = os.path.join(
            config.files.data_dir,
            "model_outputs",
            timestamp,
            config.experiment_dirs.test_output_dir,
            "".join(fp_seg_name),
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return (output_dir, fp_transcripts_processed, fp_gene_names, fp_seg, fp_seg_name)


def get_seg_map(fp_seg):
    seg_map_mi = tifffile.imread(fp_seg)
    height = seg_map_mi.shape[0]
    width = seg_map_mi.shape[1]

    cell_ids_unique = np.unique(seg_map_mi.reshape(-1))
    cell_ids_unique = cell_ids_unique[1:]
    n_cells = len(cell_ids_unique)
    print("\tNumber of cells: " + str(n_cells))

    return seg_map_mi, height, width, cell_ids_unique, n_cells


def resize_seg_map(seg_map_mi, width_pix, height_pix, output_dir, use_cv2=False):
    if use_cv2:
        import cv2
        seg_map = cv2.resize(
            seg_map_mi.astype(np.int32),
            (width_pix, height_pix),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        seg_map = resize(
            seg_map_mi.astype(np.int32),
            (height_pix, width_pix),
            order=0,  # Nearest-neighbor interpolation
            anti_aliasing=False, 
            preserve_range=True
        ).astype(np.int32)
    
    print("Segmentation map pixel size: ", seg_map.shape)
    fp_rescaled_seg = output_dir + "/rescaled.tif"
    print("Saving temporary resized segmentation")
    tifffile.imwrite(
        fp_rescaled_seg, seg_map.astype(np.uint32), photometric="minisblack"
    )

    return seg_map, fp_rescaled_seg


def make_cell_gene_mat(config: Config, is_cell: bool, timestamp: str | None = None):
    """Generates the cell-gene matrix with merged metadata."""

    print(f"Making cell gene matrix...")
    
    output_dir, fp_transcripts_processed, fp_gene_names, fp_seg, _ = get_cgm_paths(config, is_cell, timestamp)

    include_spatial = config.cgm_params.include_spatial
    x_col, y_col, gene_col = config.transcripts.x_col, config.transcripts.y_col, config.transcripts.gene_col

    seg_map_mi, height, width, cell_ids_unique, _ = get_seg_map(fp_seg)

    with open(fp_gene_names) as file:
        gene_names = [line.rstrip() for line in file]

    col_names = ["cell_id"] + gene_names

    fp_expr = os.path.join(output_dir, config.files.fp_expr)

    if not os.path.exists(fp_expr):
        print(f"Generating expression matrix...")

        df_expr = process_starmap(df_expr, get_n_processes(config.cpus), output_dir, 
                                  cell_ids_unique, col_names, seg_map_mi, x_col, y_col, gene_col)

        df_expr.to_csv(fp_expr)
        print(f"Saved expression matrix to {fp_expr}")

    else:
        print(f"Expression matrix found. Reloading...")
        df_expr = pd.read_csv(fp_expr)

    if include_spatial and is_cell:
        print("Merging metadata...")

        df_meta = process_starmap_meta(df_expr, gene_names, get_n_processes(config.cpus), output_dir, 
                                       seg_map_mi, config.affine.scale_pix_x, config.affine.scale_pix_y)

        df_merged = df_expr.merge(df_meta, on="cell_id", how="left")

        df_merged.to_csv(fp_expr, index=False)
        print(f"Updated expr_mat.csv with metadata.")

    print("Done making cell gene matrix.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")
    parser.add_argument(
        "--is_cell", type=bool, help="whether to segment cells or nuclei"
    )

    args = parser.parse_args()
    config = load_config(args.config_dir)

    make_cell_gene_mat(config, args.is_cell)
