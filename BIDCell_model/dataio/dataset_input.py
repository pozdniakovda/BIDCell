import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import sys
import os
import tifffile
import imgaug.augmenters as iaa
import h5py
import random 
import imageio
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import rotate 

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class DataProcessing(data.Dataset):
    def __init__(self, data_sources, data_params,
                 isTraining=True, all_patches=True, shift_patches=0):

        self.patch_size = data_params.patch_size 
        self.isTraining = isTraining
        
        if shift_patches == 0:
            self.expr_fp = data_sources.expr_fp + str(self.patch_size) + 'x' + str(self.patch_size)
        else:
            self.expr_fp = data_sources.expr_fp + str(self.patch_size) + 'x' + str(self.patch_size) + '_shift_' + str(shift_patches)

        self.expr_fp_ext = data_sources.expr_fp_ext

        # Check valid data directories
        if not os.path.exists(self.expr_fp):
            sys.exit("Invalid file path %s" %self.expr_fp)
        if not os.path.exists(data_sources.nuclei_fp):
            sys.exit("Invalid file path %s" %data_sources.nuclei_fp)
        if not os.path.exists(data_sources.nuclei_types_fp):
            sys.exit("Invalid file path %s" %data_sources.nuclei_types_fp)
        if not os.path.exists(data_sources.pos_markers_fp):
            sys.exit("Invalid file path %s" %data_sources.pos_markers_fp)
        if not os.path.exists(data_sources.neg_markers_fp):
            sys.exit("Invalid file path %s" %data_sources.neg_markers_fp)

        self.nuclei_fp = data_sources.nuclei_fp
        self.nuclei_types_fp = data_sources.nuclei_types_fp
            
        self.nuclei = tifffile.imread(self.nuclei_fp)
        self.nuclei = self.nuclei.astype(np.int32)
        print('Loaded nuclei')
        print(self.nuclei.shape)

        # Get coordinates of non-overlapping patches
        if shift_patches == 0:
            h_starts = list(np.arange(0, self.nuclei.shape[0]-self.patch_size, self.patch_size))
            w_starts = list(np.arange(0, self.nuclei.shape[1]-self.patch_size, self.patch_size))
            
            # Include remainder patches on 
            h_starts.append(self.nuclei.shape[0]-self.patch_size)
            w_starts.append(self.nuclei.shape[1]-self.patch_size)
        else:
            h_starts = list(np.arange(shift_patches, self.nuclei.shape[0]-self.patch_size, self.patch_size))
            w_starts = list(np.arange(shift_patches, self.nuclei.shape[1]-self.patch_size, self.patch_size))    

        coords_starts = [(x, y) for x in h_starts for y in w_starts]
        
        # Randomly select train/test samples
        random.seed(1234)
        n_coords = len(coords_starts)
        sample_ids = range(n_coords)
        sample_k = int(data_params.train_split_pct*n_coords/100)
        train_ids = random.sample(sample_ids, k=sample_k)
        
        if self.isTraining:
            self.coords_starts = [coords_starts[x] for x in train_ids] 
            self.coords_starts = self.coords_starts
        elif all_patches == True:
            self.coords_starts = coords_starts
        else:
            test_ids = [x for x in sample_ids if x not in train_ids]
            self.coords_starts = [coords_starts[x] for x in test_ids]

        # Nuclei IDs with cell types and elongated nuclei
        h5f = h5py.File(self.nuclei_types_fp, 'r')
        self.nuclei_types_idx = list(h5f['data'][:])
        self.nuclei_types_ids = list(h5f['ids'][:])
        h5f.close()
        type_names = data_params.cell_types
        types_elong = data_params.elongated
        idx_elong = [type_names.index(x) for x in types_elong]
        nuclei_types_elong = [1 if x in idx_elong else 0 for x in self.nuclei_types_idx]
        self.nuclei_ids_elong = [x for i, x in enumerate(self.nuclei_types_ids) if nuclei_types_elong[i] == 1]

        print('%d patches available' %len(self.coords_starts))

        df_pos_markers = pd.read_csv(data_sources.pos_markers_fp, index_col=0)
        df_neg_markers = pd.read_csv(data_sources.neg_markers_fp, index_col=0)

        self.pos_markers = df_pos_markers.to_numpy()
        self.neg_markers = df_neg_markers.to_numpy()
        

    def augment_data(self, batch_raw):
        batch_raw = np.expand_dims(batch_raw, 0)

        # Original, horizontal
        random_flip = np.random.randint(2, size=1)[0]
        # 0, 90, 180, 270
        random_rotate = np.random.randint(4, size=1)[0]

        # Flips
        if random_flip == 0:
            batch_flip = batch_raw*1
        else:
            batch_flip = iaa.Flipud(1.0)(images=batch_raw)
                
        # Rotations
        if random_rotate == 0:
            batch_rotate = batch_flip*1
        elif random_rotate == 1:
            batch_rotate = iaa.Rot90(1, keep_size=True)(images=batch_flip)
        elif random_rotate == 2:
            batch_rotate = iaa.Rot90(2, keep_size=True)(images=batch_flip)
        else:
            batch_rotate = iaa.Rot90(3, keep_size=True)(images=batch_flip)
        
        images_aug_array = np.array(batch_rotate)

        return images_aug_array, random_flip, random_rotate


    def normalise_images(self, imgs):        
        return (imgs - self.fold_mean)/self.fold_std


    def __len__(self):
            'Denotes the total number of samples'
            return len(self.coords_starts)


    def __getitem__(self, index):
            'Generates one sample of data'
            coords = self.coords_starts[index]
            coords_h1 = coords[0]
            coords_w1 = coords[1]
            coords_h2 = coords_h1 + self.patch_size
            coords_w2 = coords_w1 + self.patch_size

            expr_fp = self.expr_fp + '/' + str(coords_h1) + 'x' + str(coords_w1) + self.expr_fp_ext
            h5f = h5py.File(expr_fp, 'r')
            expr = h5f['data'][:].astype(np.float64)
            h5f.close()
    
            nucl = self.nuclei[coords_h1:coords_h2, coords_w1:coords_w2]
                        
            assert(expr.shape[0] == self.patch_size), print(expr.shape[0])
            assert(expr.shape[1] == self.patch_size), print(expr.shape[1])

            img = np.concatenate((expr, np.expand_dims(nucl,-1)), -1)
            
            if self.isTraining:
                img, _, _ = self.augment_data(img)           
                img = img[0,:,:,:]

            expr_aug = img[:,:,:-1] 
            nucl_aug = img[:,:,-1]

            # Tile cells individually along channel axis
            cell_ids, _ = np.unique(nucl_aug, return_index=True)
            cell_ids = cell_ids[cell_ids != 0]
            n_cells = len(cell_ids)

            nucl_split = np.zeros((self.patch_size, self.patch_size, n_cells))
            search_areas = np.zeros((self.patch_size, self.patch_size, n_cells))

            # For non-elongated cells
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
            
            # For elongated cells
            ksizevmin = 3
            ksize_total = 60
            ecc_scale = 0.9

            # For pos/neg marker loss
            search_pos = np.zeros((self.patch_size, self.patch_size, n_cells))
            search_neg = np.zeros((self.patch_size, self.patch_size, n_cells))
            kernel_posneg = np.ones((3,3), dtype=np.uint8)
                
            for i_cell, c_id in enumerate(cell_ids):
                nucl_split[:,:,i_cell] = np.where(nucl_aug == c_id, 1, 0)
               
                if c_id not in self.nuclei_ids_elong:
                    # Not elongated
                    search_areas[:,:,i_cell] = cv2.dilate(nucl_split[:,:,i_cell], kernel, iterations=1)
                else:
                    # Elongated
                    try:
                        contours = cv2.findContours(nucl_split[:,:,i_cell].astype(np.uint8), 
                                                    cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                        ellipse = cv2.fitEllipse(np.squeeze(contours[0]))
                        (center, axes, orientation) = ellipse
                        majoraxis_length = max(axes)
                        minoraxis_length = min(axes)
                        eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)

                        # Get ellipse filter based on eccentricity and majoraxis length 
                        # Rotate based on orientation
                        # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
                        ksizeh = int(round(ecc_scale*eccentricity*ksize_total))
                        ksizev = ksize_total - ksizeh if (ksize_total - ksizeh) > ksizevmin else ksizevmin
                        kernel_elong = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksizeh, ksizev))
                        kernel_elong = rotate(kernel_elong, 90-orientation, reshape=True)

                        search_areas[:,:,i_cell] = cv2.dilate(nucl_split[:,:,i_cell], kernel_elong, iterations=1)
                    except:
                        search_areas[:,:,i_cell] = cv2.dilate(nucl_split[:,:,i_cell], kernel, iterations=1)

                ct_nucleus = int(self.nuclei_types_idx[self.nuclei_types_ids.index(c_id)])

                # Markers with dilation
                ct_pos = np.expand_dims(np.expand_dims(self.pos_markers[ct_nucleus,:], 0),0)*expr_aug
                ct_pos = np.sum(ct_pos,-1)
                ct_pos[ct_pos > 0] = 1
                ct_pos[ct_pos < 0] = 0
                search_pos[:,:,i_cell] = search_areas[:,:,i_cell]*cv2.dilate(ct_pos, kernel_posneg, iterations=1)

                ct_neg = np.expand_dims(np.expand_dims(self.neg_markers[ct_nucleus,:], 0),0)*expr_aug
                ct_neg = np.sum(ct_neg,-1)
                ct_neg[ct_neg > 0] = 1
                ct_neg[ct_neg < 0] = 0
                search_neg[:,:,i_cell] = search_areas[:,:,i_cell]*cv2.dilate(ct_neg, kernel_posneg, iterations=1)

            search_areas[search_areas > 0] = 1  
            search_areas[search_areas < 0] = 0 

            expr_aug_sum = np.sum(expr_aug, -1)

            # Mask expressions and change channel order
            expr_split = np.repeat(expr_aug[:,:,:,np.newaxis], n_cells, axis=3)
            expr_split = expr_split*np.expand_dims(search_areas,2)

            # Convert to tensor
            expr_torch = torch.from_numpy(expr_split).float()
            nucl_torch = torch.from_numpy(nucl_split).long()
            search_areas_torch = torch.from_numpy(search_areas).long()
            search_pos_torch = torch.from_numpy(search_pos).long()
            search_neg_torch = torch.from_numpy(search_neg).long()
            
            if self.isTraining:
                return expr_torch, nucl_torch, search_areas_torch, search_pos_torch, search_neg_torch, coords_h1, coords_w1, nucl_aug, expr_aug_sum
            else:
                return expr_torch, nucl_torch, search_areas_torch, search_pos_torch, search_neg_torch, coords_h1, coords_w1, nucl_aug, expr_aug_sum, self.nuclei.shape[0], self.nuclei.shape[1]
