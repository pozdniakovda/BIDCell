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


def process_chunk(chunk, output_dir, cell_ids_unique, col_names, x_col, y_col, gene_col, 
                  seg_map, scale_pix_x, scale_pix_y, save_chunk=False, 
                  generate_metadata=False, assign_blank_cells=True):
    """Extract cell expression profiles"""

    # Construct the output dataframe
    if assign_blank_cells:
        # Assign rows for all cell IDs, even those that aren't present in this chunk
        df_out = pd.DataFrame(0, index=cell_ids_unique, columns=col_names)
    else:
        # Identify which cell IDs are present in this chunk
        cell_ids_in_chunk = np.unique(seg_map[chunk[y_col], chunk[x_col]])
        cell_ids_in_chunk = cell_ids_in_chunk[cell_ids_in_chunk > 0]
        df_out = pd.DataFrame(0, index=cell_ids_in_chunk, columns=col_names)
    
    df_out["cell_id"] = cell_ids_in_chunk.copy()

    chunk_id = chunk.index[0]

    for index_row, row in chunk.iterrows():
        gene = row[gene_col]
        w_loc = row[x_col]
        h_loc = row[y_col]

        seg_val = seg_map[h_loc, w_loc]
        if seg_val > 0:
            df_out.loc[seg_val, gene] += 1

    ''' Compute cell locations, sizes, eccentricity, and retrieve cell type annotations '''

    if generate_metadata:
        for cur_i, cell_id in enumerate(df_out["cell_id"]):
            if cell_id > 0:
                try:
                    # Get coordinates of the cell in segmentation mask
                    coords = np.where(seg_map_mi == cell_id)
                    x_points = coords[1]
                    y_points = coords[0]
    
                    # Compute centroid
                    df_out.at[cur_i, "cell_centroid_x"] = (sum(x_points) / len(x_points)) * scale_pix_x
                    df_out.at[cur_i, "cell_centroid_y"] = (sum(y_points) / len(y_points)) * scale_pix_y
    
                    # Compute pixel size
                    df_out.at[cur_i, "pixel_size"] = len(coords[0]) / (scale_pix_x * scale_pix_y)
    
                    # Compute eccentricity using regionprops
                    mask = (seg_map_mi == cell_id).astype(np.uint8)
                    labeled_mask = label(mask)
                    props = regionprops(labeled_mask)
                    df_out.at[cur_i, "eccentricity"] = props[0].eccentricity if props else -1
    
                except Exception:
                    df_out.at[cur_i, "cell_centroid_x"] = -1
                    df_out.at[cur_i, "cell_centroid_y"] = -1
                    df_out.at[cur_i, "pixel_size"] = -1
                    df_out.at[cur_i, "eccentricity"] = -1
            else: 
                print(f"Caution: at step #{cur_i}, current cell_id ({cell_id}) <= 0 "
                      "and will be skipped.")

        # Reorder columns
        cols = list(col_names).copy()
        cols.remove("cell_id")
        cols = ["cell_id", "cell_centroid_x", "cell_centroid_y", "pixel_size", "eccentricity"] + cols
        df_out = df_out[cols]

    # Optionally save as CSV
    if save_chunk:
        file_path = output_dir + "/" + "chunk_%d.csv" % chunk_id
        df_out.to_csv(file_path)
    else:
        file_path = None

    return (df_out, file_path)


def process_parallel(df_expr, n_processes, output_dir, cell_ids_unique, col_names, x_col, y_col, 
                     gene_col, seg_map, scale_pix_x, scale_pix_y, save_chunks=False, generate_metadata=False):
    """Parallelized CGM data processing using `starmap`; optionally calculates metadata."""
    
    df_expr_splits = np.array_split(df_expr, n_processes)
    print(f"df_expr len={len(df_expr)}")
    print(f"df_expr_splits lens: {[len(df_expr_split) for df_expr_split in df_expr_splits]}")
    results = []

    print("Extracting cell-gene matrix chunks")

    with mp.Pool(n_processes) as pool:
        args = []
        for chunk in df_expr_splits:
            args.append((chunk, output_dir, cell_ids_unique, col_names, x_col, y_col, gene_col, 
                         seg_map, scale_pix_x, scale_pix_y, save_chunks, generate_metadata))

        for i, (df_out, file_path) in enumerate(pool.starmap(process_chunk, args)):
            if file_path is not None:
                print(f"Processed and saved results for chunk #{i+1}: {file_path}")
            else:
                print(f"Processed results for chunk #{i+1}")
            
            results.append(df_out)

    # Concatenate all DataFrames
    df_merged = pd.concat(results, ignore_index=True)
    df_merged = df_merged.groupby("cell_id", as_index=False).sum()
    df_merged.set_index("cell_id", drop=True)

    return df_merged


def process_chunk_meta(matrix, fp_output, seg_map_mi, col_names_coords, 
                       scale_pix_x, scale_pix_y, cell_annotations=None, save_chunk=False):
    """Compute cell locations, sizes, eccentricity, and retrieve cell type annotations."""

    chunk_id = matrix[0, 0]
    
    # Convert to Pandas DataFrame to handle mixed data types
    df_output = pd.DataFrame(columns=col_names_coords)
    df_output["cell_id"] = matrix[:, 0].astype(int)  # Ensure IDs are integers

    # Convert to pixel resolution
    for cur_i, cell_id in enumerate(df_output["cell_id"]):
        if cell_id > 0:
            try:
                # Get coordinates of the cell in segmentation mask
                coords = np.where(seg_map_mi == cell_id)
                x_points = coords[1]
                y_points = coords[0]

                # Compute centroid
                df_output.at[cur_i, "cell_centroid_x"] = (sum(x_points) / len(x_points)) * scale_pix_x
                df_output.at[cur_i, "cell_centroid_y"] = (sum(y_points) / len(y_points)) * scale_pix_y

                # Compute pixel size
                df_output.at[cur_i, "pixel_size"] = len(coords[0]) / (scale_pix_x * scale_pix_y)

                # Compute eccentricity using regionprops
                mask = (seg_map_mi == cell_id).astype(np.uint8)
                labeled_mask = label(mask)
                props = regionprops(labeled_mask)
                df_output.at[cur_i, "eccentricity"] = props[0].eccentricity if props else -1

            except Exception:
                df_output.at[cur_i, "cell_centroid_x"] = -1
                df_output.at[cur_i, "cell_centroid_y"] = -1
                df_output.at[cur_i, "pixel_size"] = -1
                df_output.at[cur_i, "eccentricity"] = -1

    # Save as CSV
    if save_chunk:
        file_path = f"{fp_output}{chunk_id}.csv"
        df_output.to_csv(file_path, index=False)
    else:
        file_path = None

    return (df_output, file_path)


def process_parallel_meta(df_out, gene_names, n_processes, output_dir, seg_map_mi, 
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
    ]

    results = []
    
    with mp.Pool(n_processes) as pool:
        args = []
        for chunk in matrix_all_splits:
            args.append((chunk, output_dir, seg_map_mi, col_names_coords, scale_pix_x, scale_pix_y, 
                         cell_annotations, save_chunks))

        for i, (df_out, file_path) in enumerate(pool.starmap(process_chunk_meta, args)):
            if file_path is not None:
                print(f"Processed and saved meta results for chunk #{i+1}: {file_path}")
            else:
                print(f"Processed meta results for chunk #{i+1}")
            
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

    fp_expr = os.path.join(output_dir, config.files.fp_expr)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return (output_dir, fp_transcripts_processed, fp_gene_names, fp_seg, fp_seg_name, fp_expr)


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
    """Generates the cell-gene matrix but does NOT merge annotations (handled in predict())."""

    print(f"Making cell gene matrix...")    
    cgm_paths = get_cgm_paths(config, is_cell, timestamp)
    output_dir, fp_transcripts_processed, fp_gene_names, fp_seg, fp_seg_name, fp_expr = cgm_paths

    print(f"\toutput_dir: {output_dir}")

    include_spatial = config.cgm_params.include_spatial

    # Column names in the transcripts csv
    x_col = config.transcripts.x_col
    y_col = config.transcripts.y_col
    gene_col = config.transcripts.gene_col

    # Get segmentation map and associated metrics
    seg_map_mi, height, width, cell_ids_unique, n_cells = get_seg_map(fp_seg)

    # Read gene names and get cols
    with open(fp_gene_names) as file:
        gene_names = [line.rstrip() for line in file]

    col_names = ["cell_id"] + gene_names

    n_processes = get_n_processes(config.cpus)

    scale_pix_x = config.affine.scale_pix_x
    scale_pix_y = config.affine.scale_pix_y

    if not os.path.exists(fp_expr):
        print(f"\tFile does not exist; generating...")
        # Rescale to pixel size
        height_pix = np.round(height / config.affine.scale_pix_y).astype(int)
        width_pix = np.round(width / config.affine.scale_pix_x).astype(int)

        seg_map, fp_rescaled_seg = resize_seg_map(seg_map_mi, width_pix, height_pix, output_dir, use_cv2=False)

        #df_out = pd.DataFrame(0, index=cell_ids_unique, columns=col_names)
        #df_out["cell_id"] = cell_ids_unique.copy()

        # Divide into patches for large datasets that exceed memory capacity
        h_coords, _ = get_patches_coords(height_pix, config.cgm_params.max_sum_hw // 2)
        w_coords, _ = get_patches_coords(width_pix, config.cgm_params.max_sum_hw - (config.cgm_params.max_sum_hw // 2))
        hw_coords = [(hs, he, ws, we) for (hs, he) in h_coords for (ws, we) in w_coords]

        seg_map_full = tifffile.imread(fp_rescaled_seg)

        output_dfs = []
        for hs, he, ws, we in tqdm(hw_coords):
            seg_map, df_expr = prepare_expr(seg_map_full, hs, he, ws, we, fp_transcripts_processed, 
                                            x_col, y_col, scale_pix_x, scale_pix_y, print_ranges=False)

            df_expr.reset_index(drop=True, inplace=True)

            print("Extracting cell-gene matrix chunks")
            save_chunks = False
            generate_metadata = True if include_spatial and is_cell else False
            df_out = process_parallel(df_expr, n_processes, output_dir, cell_ids_unique, col_names, x_col, y_col, 
                                      gene_col, seg_map, scale_pix_x, scale_pix_y, save_chunks, False)

            fp_chunks = glob.glob(os.path.join(output_dir, "chunk_*.csv"))
            #for fpc in fp_chunks:
            #    df_i = pd.read_csv(fpc, index_col=0)
            #    df_out.iloc[:, 1:] = df_out.iloc[:, 1:].add(df_i.iloc[:, 1:])

            #output_dfs.append(df_out)
            df_out.to_csv(fp_expr)
            print(f"Saved current cell-gene matrix to {fp_expr}")

            # Clean up chunk files
            if save_chunks:
                for fpc in fp_chunks:
                    os.remove(fpc)

        print("Obtained cell-gene matrix")
        os.remove(fp_rescaled_seg)
        del seg_map, df_expr

    else:
        print(f"\tFile exists; reloading...")
        df_out = pd.read_csv(fp_expr, index_col=0)

    if include_spatial and is_cell:
        print("Computing cell locations and sizes...")
        df_meta = process_parallel_meta(df_out, gene_names, n_processes, output_dir, seg_map_mi, 
                                        scale_pix_x, scale_pix_y, cell_annotations=None, save_chunks=False)

        fp_expr_meta = fp_expr.rsplit(".", 1)[0] + "_meta.csv"
        df_meta.to_csv(fp_expr_meta)
        print(f"Saved current cell-gene matrix meta information to {fp_expr_meta}")

        # Merge the dataframes together
        df_merged = pd.merge(df_meta, df_out, on="cell_id", how="inner")
        
        # Save merged file as expr_mat.csv (overwrite existing)
        fp_expr_merged = fp_expr.rsplit(".", 1)[0] + "_merged.csv"
        df_merged.to_csv(fp_expr_merged)
        print(f"Saved merged cell-gene matrix to {fp_expr_merged}")
        print("Done making cell gene matrix; returned df_merged.")
        return (df_merged, output_dir)
    
    else: 
        print("Done making cell gene matrix; returned df_out.")
        return (df_out, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")
    parser.add_argument(
        "--is_cell", type=bool, help="whether to segment cells or nuclei"
    )

    args = parser.parse_args()
    config = load_config(args.config_dir)

    make_cell_gene_mat(config, args.is_cell)
