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


def process_fast(
    chunk, output_dir, cell_ids_unique, col_names, seg_map, x_col, y_col, gene_col
):
    """Fast extraction of cell expression profiles using NumPy-based indexing."""

    chunk_id = chunk.index[0]  # Get the chunk ID

    # Create a NumPy-based storage instead of DataFrame for efficiency
    df_out = np.zeros((len(cell_ids_unique), len(col_names)), dtype=int)

    # Mapping cell IDs to row indices in `df_out`
    cell_id_to_index = {cell_id: i for i, cell_id in enumerate(cell_ids_unique)}

    # Extract relevant columns as NumPy arrays (faster than Pandas operations)
    genes = chunk[gene_col].values
    w_locs = chunk[x_col].values
    h_locs = chunk[y_col].values

    # Vectorized mask to check valid segment locations
    seg_vals = seg_map[h_locs, w_locs]

    valid_mask = seg_vals > 0  # Boolean mask for valid assignments
    valid_seg_vals = seg_vals[valid_mask]
    valid_genes = genes[valid_mask]

    # Convert segment values to indices in `df_out`
    valid_indices = np.array([cell_id_to_index[seg] for seg in valid_seg_vals])

    # Use NumPy's efficient advanced indexing for fast accumulation
    np.add.at(df_out, (valid_indices, [col_names.index(gene) for gene in valid_genes]), 1)

    # Convert back to Pandas DataFrame before saving
    df_out_df = pd.DataFrame(df_out, index=cell_ids_unique, columns=col_names)
    df_out_df["cell_id"] = cell_ids_unique

    df_out_df.to_csv(f"{output_dir}/chunk_{chunk_id}.csv")


def process_chunk(
    chunk, output_dir, cell_ids_unique, col_names, seg_map, x_col, y_col, gene_col
):
    """Extract cell expression profiles"""

    df_out = pd.DataFrame(0, index=cell_ids_unique, columns=col_names)
    df_out["cell_id"] = cell_ids_unique.copy()

    chunk_id = chunk.index[0]

    for index_row, row in chunk.iterrows():
        gene = row[gene_col]
        w_loc = row[x_col]
        h_loc = row[y_col]

        seg_val = seg_map[h_loc, w_loc]
        if seg_val > 0:
            df_out.loc[seg_val, gene] += 1

    df_out.to_csv(output_dir + "/" + "chunk_%d.csv" % chunk_id)


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
                    seg_map, x_col, y_col, gene_col):
    # Method #2: Starmap and dedicated Pool
    df_expr_splits = np.array_split(df_expr, n_processes)
    with mp.Pool(n_processes) as pool:
        pool.starmap(
            process_chunk,
            [
                (
                    chunk,
                    output_dir,
                    cell_ids_unique,
                    col_names,
                    seg_map,
                    x_col,
                    y_col,
                    gene_col,
                )
                for chunk in df_expr_splits
            ],
        )


def process_chunk_meta(
    matrix, fp_output, seg_map_mi, col_names_coords, scale_pix_x, scale_pix_y, cell_annotations=None
):
    """Compute cell locations and sizes"""

    chunk_id = matrix[0, 0]
    output = np.zeros((matrix.shape[0], len(col_names_coords)))
    output[:, 0] = matrix[:, 0].copy()
    output[:, 4:] = matrix[:, 1:].copy()

    # Convert to pixel resolution
    for cur_i, cell_id in enumerate(output[:, 0]):
        if cell_id > 0:
            try:
                # cell_centroid_x and cell_centroid_y
                coords = np.where(seg_map_mi == cell_id)
                x_points = coords[1]
                y_points = coords[0]
                centroid_x = sum(x_points) / len(x_points)
                centroid_y = sum(y_points) / len(y_points)
                output[cur_i, 1] = centroid_x * scale_pix_x
                output[cur_i, 2] = centroid_y * scale_pix_y
    
                # cell_size (renamed to pixel_size)
                pixel_size = len(coords[0]) / (scale_pix_x * scale_pix_y)
                output[cur_i, 3] = pixel_size
    
                # Compute eccentricity using regionprops
                mask = (seg_map_mi == cell_id).astype(np.uint8)
                labeled_mask = label(mask)
                props = regionprops(labeled_mask)
                eccentricity = props[0].eccentricity if props else -1
                output[cur_i, 4] = eccentricity  # Assign eccentricity
    
                # Assign cell_type, spearman, and cell_type_atlas
                if cell_annotations is not None and cell_id in cell_annotations:
                    annotation = cell_annotations[cell_id]
                    output[cur_i, 5] = annotation.get("cell_type", "Unknown")
                    output[cur_i, 6] = annotation.get("spearman", -1)
                    output[cur_i, 7] = annotation.get("cell_type_atlas", "Unknown")
                else:
                    output[cur_i, 5:8] = ["Unknown", -1, "Unknown"]  # Default values
    
            except Exception:
                output[cur_i, 1:8] = [-1, -1, -1, -1, "Unknown", -1, "Unknown"]  # Fill with defaults on error

    # Save as csv
    col_names_coords = [
        "cell_id",
        "cell_centroid_x",
        "cell_centroid_y",
        "pixel_size",  # Renamed from cell_size
        "eccentricity",
        "cell_type",
        "spearman",
        "cell_type_atlas"
    ]
    
    df_split = pd.DataFrame(
        output, index=list(range(output.shape[0])), columns=col_names_coords
    )
    df_split.to_csv(fp_output + "%d.csv" % chunk_id, index=False)


def process_starmap_meta(df_expr, n_processes, output_dir, seg_map_mi, col_names_coords, 
                         scale_pix_x, scale_pix_y, cell_annotations):
    # Method #2: Starmap and dedicated Pool for meta
    df_expr_splits = np.array_split(df_expr, n_processes)
    with mp.Pool(n_processes) as pool:
        pool.starmap(
            process_chunk_meta,  # Use process_chunk_meta instead of process_chunk
            [
                (
                    chunk,
                    output_dir,
                    seg_map_mi,  # Segmentation map
                    col_names_coords,  # Updated column names
                    scale_pix_x,
                    scale_pix_y,
                    cell_annotations,  # Pass cell annotations dictionary
                )
                for chunk in df_expr_splits
            ],
        )


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
    print("Number of cells " + str(n_cells))

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
    # Generates the cell gene matrix

    print(f"Making cell gene matrix...")
    t0 = time.time()
    
    cgm_paths = get_cgm_paths(config, is_cell, timestamp)
    output_dir, fp_transcripts_processed, fp_gene_names, fp_seg, fp_seg_name = cgm_paths

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
    # print(f"Number of splits for multiprocessing: {n_processes}")

    '''
    Scale factor to pixel resolution of platform
    read in affine
    extract scale_x and scale_y
    divide by (scale_x*pixel resolution) (microns per pixel)
    affine = pd.read_csv(fp_affine, index_col=0, header=None, sep='\t')
    scale_x_tr = float(affine.loc["scale_x"].item())
    scale_y_tr = float(affine.loc["scale_y"].item())
    scale_pix_x = (scale_x_tr*config.affine.scale_pix_x)
    scale_pix_y = (scale_y_tr*config.affine.scale_pix_y)
    '''
    scale_pix_x = config.affine.scale_pix_x
    scale_pix_y = config.affine.scale_pix_y

    t1 = time.time()
    print(f"\tInitialization: {t1-t0} seconds")

    if not os.path.exists(output_dir + "/" + config.files.fp_expr):
        # Rescale to pixel size
        height_pix = np.round(height / config.affine.scale_pix_y).astype(int)
        width_pix = np.round(width / config.affine.scale_pix_x).astype(int)

        seg_map, fp_rescaled_seg = resize_seg_map(seg_map_mi, width_pix, height_pix, output_dir, use_cv2=False)

        t2 = time.time()
        print(f"\tSegmentation map resizing: {t2-t1} seconds")

        df_out = pd.DataFrame(0, index=cell_ids_unique, columns=col_names)
        df_out["cell_id"] = cell_ids_unique.copy()

        t3 = time.time()
        print(f"\tOutput dataframe generation: {t3-t2} seconds")

        # Divide into patches for large datasets that exceed memory capacity
        if (height_pix + width_pix) > config.cgm_params.max_sum_hw:
            patch_h = int(config.cgm_params.max_sum_hw / 2)
            patch_w = config.cgm_params.max_sum_hw - patch_h
        else:
            patch_h = height_pix
            patch_w = width_pix

        h_coords, _ = get_patches_coords(height_pix, patch_h)
        w_coords, _ = get_patches_coords(width_pix, patch_w)
        hw_coords = [(hs, he, ws, we) for (hs, he) in h_coords for (ws, we) in w_coords]

        t4 = time.time()
        print(f"\tGetting patches coords: {t4-t3} seconds")

        #print("Extracting cell expressions")
        seg_map_full = tifffile.imread(fp_rescaled_seg)

        t5 = time.time()
        print(f"\tLoading full segmentation map: {t5-t4} seconds")
        
        for hs, he, ws, we in tqdm(hw_coords):
            t6 = time.time()
            # Prepare expression data and segmentation map subset for processing
            seg_map, df_expr = prepare_expr(seg_map_full, hs, he, ws, we, fp_transcripts_processed, 
                                            x_col, y_col, scale_pix_x, scale_pix_y, print_ranges=False)
            t7 = time.time()
            print(f"\Preparing data subset for processing: {t7-t6} seconds")

            df_expr.reset_index(drop=True, inplace=True)

            print("Extracting cell-gene matrix chunks")
            processes = []

            # Method #1: Pass the whole dataset instead of chunks
            #process_fast(df_expr, output_dir, cell_ids_unique, col_names, seg_map,
            #             x_col, y_col, gene_col)
            
            # Method #2: Starmap and dedicated Pool
            process_starmap(df_expr, n_processes, output_dir, cell_ids_unique, col_names, 
                            seg_map, x_col, y_col, gene_col)
            
            # Method #3: Original
            #process_parallel(df_expr, n_processes, output_dir, cell_ids_unique, col_names, 
            #                 seg_map, x_col, y_col, gene_col)

            t8 = time.time()
            print(f"\Processing data: {t8-t7} seconds")

            #print("Combining cell-gene matrix chunks")

            fp_chunks = glob.glob(output_dir + "/chunk_*.csv")
            for fpc in fp_chunks:
                df_i = pd.read_csv(fpc, index_col=0)
                df_out.iloc[:, 1:] = df_out.iloc[:, 1:].add(df_i.iloc[:, 1:])

            df_out.to_csv(output_dir + "/" + config.files.fp_expr)

            t9 = time.time()
            print(f"\tCombining cell gene matrix chunks: {t10-t9} seconds")

            # Clean up
            for fpc in fp_chunks:
                os.remove(fpc)

            t10 = time.time()
            print(f"\tCleanup: {t10-t9} seconds")

        print("Obtained cell-gene matrix")
        
        os.remove(fp_rescaled_seg)
        del seg_map
        del df_expr

    else:
        df_out = pd.read_csv(output_dir + "/" + config.files.fp_expr, index_col=0)

    if include_spatial:
        if is_cell:
            print("Computing cell locations and sizes...")

            process_starmap_meta(df_expr, n_processes, output_dir, seg_map_mi, col_names_coords, 
                                 scale_pix_x, scale_pix_y, cell_annotations)
    
            matrix_all = df_out.to_numpy().astype(np.float32)
            matrix_all_splits = np.array_split(matrix_all, n_processes)
            processes = []
    
            fp_output = output_dir + "/cell_outputs_"
            col_names_coords = [
                "cell_id",
                "cell_centroid_x",
                "cell_centroid_y",
                "cell_size",
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

    print("Done making cell gene matrix.")

    # print("Cleaning up...")
    # mp.active_children()
    # mp.pool = None
    # print("\tDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, help="path to config")
    parser.add_argument(
        "--is_cell", type=bool, help="whether to segment cells or nuclei"
    )

    args = parser.parse_args()
    config = load_config(args.config_dir)

    make_cell_gene_mat(config, args.is_cell)
