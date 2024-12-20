import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


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
    Penalize double (or more) assignments of transcripts due to overlapping segmentations.
    """

    def __init__(self, weight, device):
        super(MultipleAssignmentLoss, self).__init__()
        self.weight = weight
        self.init_weight = weight
        self.device = device

    def forward(self, seg_pred, transcript_map, weight=None):
        """
        Args:
            seg_pred: Segmentation predictions (batch_size, num_classes, height, width).
            transcript_map: Binary map of transcript locations (batch_size, 1, height, width),
                            where 1 indicates a transcript's presence at that pixel.
            weight: Optional weight to override the initialized weight.
        """
        if weight is not None:
            self.weight = weight

        # Apply softmax to get probabilities for each pixel
        seg_probs = F.softmax(seg_pred, dim=1)

        # Extract the "cell" probabilities (assuming class 1 corresponds to cells)
        cell_probs = seg_probs[:, 1, :, :]  # (batch_size, height, width)

        # Mask cell probabilities by transcript locations
        assigned_probs = cell_probs * transcript_map[:, 0, :, :]  # (batch_size, height, width)

        # Sum probabilities for each transcript location (overlapping predictions add up)
        summed_probs = torch.sum(assigned_probs, dim=(1, 2))  # (batch_size)

        # Penalize probabilities > 1 (indicating multiple assignments)
        multiple_assignment_penalty = F.relu(summed_probs - 1)

        # Compute the total penalty normalized by the batch size
        loss = torch.mean(multiple_assignment_penalty) * self.weight

        return loss
