import torch
import torch.nn as nn
from .losses import NucleiEncapsulationLoss, OversegmentationLoss, CellCallingLoss, OverlapLoss, PosNegMarkerLoss


class NucEncapOverlapLoss(nn.Module):
    """
    Combines NucleiEncapsulationLoss and OverlapLoss.
    This loss ensures that nuclei are encapsulated within predicted cells and penalizes overlaps between different cells.
    """

    def __init__(self, weight_encap, weight_overlap, device) -> None:
        super(NucEncapOverlapLoss, self).__init__()
        self.nuclei_encapsulation_loss = NucleiEncapsulationLoss(weight_encap, device)
        self.overlap_loss = OverlapLoss(weight_overlap, device)

    def forward(self, seg_pred, batch_n, weight_encap=None, weight_overlap=None, distance_scaling=False, intensity_weighting=False):
        # Compute NucleiEncapsulationLoss
        encap_loss = self.nuclei_encapsulation_loss(seg_pred, batch_n, weight=weight_encap)

        # Compute OverlapLoss
        overlap_loss = self.overlap_loss(seg_pred, batch_n, weight=weight_overlap, 
                                         distance_scaling=distance_scaling, intensity_weighting=intensity_weighting)

        # Return the sum of both losses
        return encap_loss + overlap_loss


class OversegOverlapLoss(nn.Module):
    """
    Combines OversegmentationLoss and OverlapLoss.
    This loss minimizes oversegmentation and penalizes overlaps between different cells.
    """

    def __init__(self, weight_oversegmentation, weight_overlap, device) -> None:
        super(OversegOverlapLoss, self).__init__()
        self.oversegmentation_loss = OversegmentationLoss(weight_oversegmentation, device)
        self.overlap_loss = OverlapLoss(weight_overlap, device)

    def forward(self, seg_pred, batch_n, weight_oversegmentation=None, weight_overlap=None, 
                distance_scaling=False, intensity_weighting=False):
        # Compute OversegmentationLoss
        oversegmentation_loss = self.oversegmentation_loss(seg_pred, batch_n, weight=weight_oversegmentation)

        # Compute OverlapLoss
        overlap_loss = self.overlap_loss(seg_pred, batch_n, weight=weight_overlap, distance_scaling=distance_scaling, 
                                         intensity_weighting=intensity_weighting)

        # Return the sum of both losses
        return oversegmentation_loss + overlap_loss


class CellCallingMarkerLoss(nn.Module):
    """
    Combines CellCallingLoss and PosNegMarkerLoss.
    This loss maximizes assignment of transcripts to cells and handles positive/negative markers of cell types.
    """

    def __init__(self, cc_weight, weight_pos, weight_neg, device) -> None:
        super(CellCallingMarkerLoss, self).__init__()
        self.cell_calling_loss = CellCallingLoss(cc_weight, device)
        self.pos_neg_marker_loss = PosNegMarkerLoss(weight_pos, weight_neg, device)

    def forward(self, seg_pred, batch_sa, batch_pos, batch_neg, cc_weight=None, weight_pos=None, weight_neg=None):
        # Compute CellCallingLoss
        calling_loss = self.cell_calling_loss(
            seg_pred, batch_sa, weight=cc_weight
        )

        # Compute PosNegMarkerLoss
        marker_loss = self.pos_neg_marker_loss(
            seg_pred, batch_pos, batch_neg, weight_pos=weight_pos, weight_neg=weight_neg
        )

        # Return the sum of both losses
        return calling_loss + marker_loss
