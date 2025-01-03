import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class NucleiEncapsulationLoss(nn.Module):
    """
    Ensure that nuclei are fully within predicted cells
    """

    def __init__(self, weight, device) -> None:
        super(NucleiEncapsulationLoss, self).__init__()
        self.weight = weight
        self.init_weight = weight
        self.device = device

    def forward(self, seg_pred, batch_n, weight=None):
        # Overwrite self.weight if new weight is given; original is preserved as self.init_weight
        if weight is not None:
            self.weight = weight

        criterion_ce = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion_ce(seg_pred, batch_n[:, 0, :, :])

        return self.weight * loss


class OversegmentationLoss(nn.Module):
    """
    Minimise oversegmentation
    """

    def __init__(self, weight, device) -> None:
        super(OversegmentationLoss, self).__init__()
        self.weight = weight
        self.init_weight = weight
        self.device = device
    
    def forward(self, seg_pred, batch_n, weight=None, alpha=1.0):
        # Overwrite self.weight if new weight is given; original is preserved as self.init_weight
        if weight is not None:
            self.weight = weight
    
        # Use the ground truth nuclei mask
        batch_n = batch_n[:, 0, :, :]
    
        # Compute class probabilities using softmax
        seg_probs = torch.nn.functional.softmax(seg_pred, dim=1)
    
        # Nuclei predictions masked by the ground truth
        probs_nuc = seg_probs[:, 1, :, :] * batch_n
    
        # Cytoplasm predictions masked by non-nuclei regions
        mask_cyto = torch.ones(batch_n.shape).to(self.device) - batch_n
        probs_cyto = seg_probs[:, 1, :, :] * mask_cyto
    
        # Apply sigmoid to emphasize probabilities > 0.5
        preds_nuc = torch.sigmoid((probs_nuc - 0.5) * alpha)
        preds_cyto = torch.sigmoid((probs_cyto - 0.5) * alpha)

        count_nuc = torch.sum(preds_nuc)
        count_cyto = torch.sum(preds_cyto)
    
        # Compute extra cytoplasm predictions (oversegmentation)
        extra = count_cyto - count_nuc
        m = torch.nn.ReLU()
        loss = m(extra)
    
        # Normalize by batch size and apply weight
        loss = (loss / seg_pred.shape[0]) * self.weight
    
        return loss


class CellCallingLoss(nn.Module):
    """
    Maximise assignment of transcripts to cells
    """

    def __init__(self, weight, device) -> None:
        super(CellCallingLoss, self).__init__()
        self.weight = weight
        self.init_weight = weight
        self.device = device

    def forward(self, seg_pred, batch_sa, weight=None):
        # Overwrite self.weight if new weight is given; original is preserved as self.init_weight
        if weight is not None:
            self.weight = weight
        
        # Limit to searchable area where there is detected expression
        penalisable = batch_sa * 1
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="none")
        loss = criterion_ce(seg_pred, penalisable[:, 0, :, :])

        loss_total = torch.sum(loss)
        loss_total = (loss_total / seg_pred.shape[0]) * self.weight

        return loss_total


class OverlapLoss(nn.Module):
    """
    Penalize overlaps between different cells. Optionally uses distance-based scaling and intensity weighting.
    """

    def __init__(self, weight, device) -> None:
        super(OverlapLoss, self).__init__()
        self.weight = weight
        self.init_weight = weight
        self.device = device

    def forward(self, seg_pred, batch_n, weight=None, alpha=1.0):
        # Overwrite self.weight if new weight is given; original is preserved as self.init_weight
        if weight is not None:
            self.weight = weight

        batch_n = batch_n[:, 0, :, :]  # Extract nuclei segmentation
        seg_probs = F.softmax(seg_pred, dim=1)

        # Combine all nuclei into a single binary mask
        all_nuclei = torch.sum(batch_n, 0)
        all_not_nuclei = torch.ones(batch_n.shape).to(self.device) - all_nuclei

        # Cytoplasm probabilities and predictions where there are no nuclei
        probs_cyto = seg_probs[:, 1, :, :] * all_not_nuclei
        preds_cyto = torch.sigmoid((probs_cyto - 0.5) * alpha)

        # Calculate the overlap (regions where multiple cell probabilities exceed 0.5)
        count_cyto_overlap = torch.sum(preds_cyto, 0) - all_not_nuclei
        m = torch.nn.ReLU()
        count_cyto_overlap = m(count_cyto_overlap)
        
        # Compute final loss
        loss = torch.sum(count_cyto_overlap)
        scale = seg_pred.shape[0] * seg_pred.shape[2] * seg_pred.shape[3]
        loss = (loss / scale) * self.weight

        return loss


class PosNegMarkerLoss(nn.Module):
    """
    Positive and negative markers of cell type
    """

    def __init__(self, weight_pos, weight_neg, device) -> None:
        super(PosNegMarkerLoss, self).__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.init_weight_pos = weight_pos
        self.init_weight_neg = weight_neg
        self.device = device

    def forward(self, seg_pred, batch_pos, batch_neg, weight_pos=None, weight_neg=None, alpha=1.0):
        # Overwrite weights if new weights are given; originals are preserved as self.init_weight_pos or _neg
        if weight_pos is not None:
            self.weight_pos = weight_pos
        if weight_neg is not None:
            self.weight_neg = weight_neg
        
        batch_pos = batch_pos[:, 0, :, :]
        batch_neg = batch_neg[:, 0, :, :]

        # POSITIVE markers
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="sum")
        loss_pos = criterion_ce(seg_pred, batch_pos)

        # NEGATIVE markers
        seg_probs = F.softmax(seg_pred, dim=1)
        probs_cells = seg_probs[:, 1, :, :]
        preds_cells = torch.sigmoid((probs_cells - 0.5) * alpha)

        loss_neg = torch.sum(preds_cells * batch_neg)

        # Calculate total loss
        loss_total = (loss_pos * self.weight_pos) + (loss_neg * self.weight_neg)
        loss_total = loss_total / seg_pred.shape[0]

        return loss_total


class MultipleAssignmentLoss(nn.Module):
    """
    Penalize assigning transcripts to more than one cell.
    """

    def __init__(self, weight, device) -> None:
        super(MultipleAssignmentLoss, self).__init__()
        self.weight = weight
        self.init_weight = weight
        self.device = device

    def forward(self, seg_pred, batch_expr_sum, weight=None, alpha=1.0, verbose=True):
        '''
        Forward pass

        Args:
            seg_pred:       predicted binary cell segmentations; shape: [n_cells, 2, H, W]
            batch_expr_sum: summed expression map;               shape: [1, 1, H, W]
            weight:         overriding weight
            alpha:          controls sharpness of sigmoid function
            verbose:        whether to display tensor shapes

        Returns: 
            loss:           multiple assignment loss
        '''
        
        # Overwrite self.weight if new weight is given; original is preserved as self.init_weight
        if weight is not None:
            self.weight = weight

        # Compute softmax probabilities
        seg_probs = F.softmax(seg_pred, dim=1)
        probs_cell = seg_probs[:, 1, :, :]  # Class 1 corresponds to cells; new shape is [n_cells, H, W]
        preds_cyto = torch.sigmoid((probs_cell - 0.5) * alpha) # emphasize probabilities > 0.5; shape: [n_cells, H, W]

        # Sum over all cells to get the total number of assignments per pixel
        total_cell_assignments = torch.sum(preds_cyto, dim=0) # shape: [H, W]

        # Penalize pixels assigned to more than one cell
        extra_assignments = torch.clamp(total_cell_assignments - 1, min=0) # shape: [H, W]

        # Mask with expression data (penalize only for pixels with expression)
        batch_expr_sum = batch_expr_sum.squeeze() # results in shape: [H, W]
        penalty = extra_assignments * batch_expr_sum  # shape: [H, W]

        # Sum the penalty over all pixels and normalize by batch size
        loss = torch.sum(penalty) # results in float value
        loss = (loss / seg_pred.shape[0]) * self.weight

        if verbose: 
            print(f"Multiple Assignment Loss argument shapes:\n"
                  f"\tseg_pred shape: {seg_pred.shape}\n"
                  f"\tbatch_expr_sum (expr_aug_sum) shape: {batch_expr_sum.shape}\n"
                  f"\tseg_probs shape: {seg_probs.shape}\n"
                  f"\tprobs_cell shape: {probs_cell.shape}\n"
                  f"\tpreds_cyto shape: {preds_cyto.shape}\n"
                  f"\ttotal_cell_assignments shape: {total_cell_assignments.shape}\n"
                  f"\textra_assignments shape: {extra_assignments.shape}\n"
                  f"\tpenalty shape: {penalty.shape}\n"
                  f"\tloss: {loss}")
        
        return loss
