import torch
import torch.nn as nn


class NucEncapOverlapLoss(nn.Module):
    """
    Ensure that nuclei are fully within predicted cells AND penalise overlaps between different cells
    """

    def __init__(self, ne_weight, ov_weight, device) -> None:
        super(NucEncapOverlapLoss, self).__init__()
        self.ne_weight = ne_weight
        self.ov_weight = ov_weight
        self.device = device

    def forward(self, seg_pred, batch_n, ne_weight=None, ov_weight=None):
        # Optionally override weights
        if ne_weight is None:
            ne_weight = self.ne_weight
        if ov_weight is None:
            ov_weight = self.ov_weight
        
        # Nuclei encapsulation loss
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="mean")
        ne_loss = criterion_ce(seg_pred, batch_n[:, 0, :, :])
        weighted_ne_loss = ne_weight * ne_loss

        # Overlap loss
        batch_n = batch_n[:, 0, :, :]
        seg_probs = torch.nn.functional.softmax(seg_pred, dim=1)

        all_nuclei = torch.sum(batch_n, 0)
        all_not_nuclei = torch.ones(batch_n.shape).to(self.device) - all_nuclei

        probs_cyto = seg_probs[:, 1, :, :] * all_not_nuclei

        ones = torch.ones(probs_cyto.shape).to(self.device)
        zeros = torch.zeros(probs_cyto.shape).to(self.device)

        # Penalise overlap loss if number of cell prob > 0.5 is > 1
        alpha = 1.0
        preds_cyto = torch.sigmoid((probs_cyto - 0.5) * alpha) * (ones - zeros) + zeros
        count_cyto_overlap = torch.sum(preds_cyto, 0) - all_not_nuclei
        m = torch.nn.ReLU()
        count_cyto_overlap = m(count_cyto_overlap)

        ov_loss = torch.sum(count_cyto_overlap)

        scale = seg_pred.shape[0] * seg_pred.shape[2] * seg_pred.shape[3]
        ov_loss = ov_loss / scale
        weighted_ov_loss = ov_weight * ov_loss

        combined_weighted_loss = weighted_ne_loss + weighted_ov_loss
        return combined_weighted_loss

    def get_max(self, seg_pred, batch_n, ne_weight=None, ov_weight=None):
        # Calculate the maximum possible value of NucEncapOverlapLoss.

        # Determine the shape parameters
        N, C, H, W = seg_pred.shape
    
        # Calculate the maximum CrossEntropy loss per pixel for NE loss
        ne_weight = self.ne_weight if ne_weight is None else ne_weight
        max_ce_loss = torch.log(torch.tensor(C, dtype=torch.float32))
        max_ne_loss = ne_weight * (max_ce_loss / N)
    
        # Calculate the non-nuclei area for OV loss
        all_nuclei = torch.sum(batch_n[:, 0, :, :], 0)
        all_not_nuclei = torch.ones_like(all_nuclei) - all_nuclei
    
        # Calculate the maximum overlap loss
        ov_weight = self.ov_weight if ov_weight is None else ov_weight
        max_ov_loss = ov_weight * (torch.sum(all_not_nuclei) / (N * H * W))
    
        # Calculate the maximum combined loss
        max_combined_loss = max_ne_loss + max_ov_loss
    
        return max_combined_loss.item()


class CellCallingMarkerLoss(nn.Module):
    """
    Maximise assignment of transcripts to cells
    """

    def __init__(self, cc_weight, pos_weight, neg_weight, device) -> None:
        super(CellCallingMarkerLoss, self).__init__()
        self.cc_weight = cc_weight
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.device = device

    def forward(self, seg_pred, batch_sa, batch_pos, batch_neg, cc_weight=None, pos_weight=None, neg_weight=None):
        # If no override weights are given, use the defaults
        if cc_weight is None:
            cc_weight = self.cc_weight
        if pos_weight is None:
            pos_weight = self.pos_weight
        if neg_weight is None: 
            neg_weight = self.neg_weight
        
        # Limit to searchable area where there is detected expression
        penalisable = batch_sa * 1
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="none")
        loss = criterion_ce(seg_pred, penalisable[:, 0, :, :])

        cc_loss_total = torch.sum(loss)
        cc_loss_total = cc_loss_total / seg_pred.shape[0]
        weighted_cc_total = cc_weight * cc_loss_total

        # Positive/negative marker losses
        batch_pos = batch_pos[:, 0, :, :]
        batch_neg = batch_neg[:, 0, :, :]

        # POSITIVE markers
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="sum")
        loss_pos = criterion_ce(seg_pred, batch_pos)

        # NEGATIVE markers
        seg_probs = torch.nn.functional.softmax(seg_pred, dim=1)
        probs_cells = seg_probs[:, 1, :, :]

        ones = torch.ones(probs_cells.shape).to(self.device)
        zeros = torch.zeros(probs_cells.shape).to(self.device)

        alpha = 1.0

        preds_cells = (torch.sigmoid((probs_cells - 0.5) * alpha) * (ones - zeros) + zeros)

        loss_neg = torch.sum(preds_cells * batch_neg)

        weighted_pn_total = (pos_weight * loss_pos + neg_weight * loss_neg) / seg_pred.shape[0]

        # Pull it all together
        combined_weighted_loss = weighted_cc_total + weighted_pn_total
        
        return combined_weighted_loss

    def get_max(self, seg_pred, batch_sa, batch_pos, batch_neg, cc_weight=None, pos_weight=None, neg_weight=None):
        # Calculate the maximum possible value of CellCallingMarkerLoss.

        # Determine the shape parameters
        N, C, H, W = seg_pred.shape
    
        # Calculate the maximum CrossEntropy loss per pixel
        max_ce_loss = torch.log(torch.tensor(C, dtype=torch.float32))
    
        # Calculate the maximum possible CC loss
        total_penalisable_pixels = torch.sum(batch_sa[:, 0, :, :])
        max_cc_loss = cc_weight * (max_ce_loss * total_penalisable_pixels / N)
    
        # Calculate the maximum possible POS loss
        total_pos_pixels = torch.sum(batch_pos[:, 0, :, :])
        max_pos_loss = pos_weight * (max_ce_loss * total_pos_pixels)
    
        # Calculate the maximum possible NEG loss
        total_neg_pixels = torch.sum(batch_neg[:, 0, :, :])
        max_neg_loss = neg_weight * total_neg_pixels
    
        # Calculate the maximum combined PN loss
        max_pn_loss = (max_pos_loss + max_neg_loss) / N
    
        # Calculate the maximum combined total loss
        max_combined_loss = max_cc_loss + max_pn_loss
    
        return max_combined_loss.item()


class NucleiEncapsulationLoss(nn.Module):
    """
    Ensure that nuclei are fully within predicted cells
    """

    def __init__(self, weight, device) -> None:
        super(NucleiEncapsulationLoss, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, seg_pred, batch_n, weight=None):
        if weight is None:
            weight = self.weight
        
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion_ce(seg_pred, batch_n[:, 0, :, :])

        return weight * loss

    def get_max(self, seg_pred, weight=None):
    # Calculate the maximum possible loss value

    weight = self.weight if weight is None else weight
    C = seg_pred.shape[1]
    max_ce_loss = torch.log(torch.tensor(C, dtype=torch.float32))
    max_loss = weight * max_ce_loss.item()
    
    return max_loss


class Oversegmentation(nn.Module):
    """
    Minimise oversegmentation
    """

    def __init__(self, weight, device) -> None:
        super(Oversegmentation, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, seg_pred, batch_n, weight=None):
        if weight is None:
            weight = self.weight
        
        batch_n = batch_n[:, 0, :, :]

        seg_probs = torch.nn.functional.softmax(seg_pred, dim=1)
        probs_nuc = seg_probs[:, 1, :, :] * batch_n

        mask_cyto = torch.ones(batch_n.shape).to(self.device) - batch_n
        probs_cyto = seg_probs[:, 1, :, :] * mask_cyto

        ones = torch.ones(probs_cyto.shape).to(self.device)
        zeros = torch.zeros(probs_cyto.shape).to(self.device)

        alpha = 1.0

        preds_nuc = torch.sigmoid((probs_nuc - 0.5) * alpha) * (ones - zeros) + zeros
        count_nuc = torch.sum(preds_nuc)

        preds_cyto = torch.sigmoid((probs_cyto - 0.5) * alpha) * (ones - zeros) + zeros
        count_cyto = torch.sum(preds_cyto)

        extra = count_cyto - count_nuc
        m = torch.nn.ReLU()
        loss = m(extra)

        loss = loss / seg_pred.shape[0]

        return weight * loss
    
    def get_max(self, seg_pred, batch_n, weight=None):
        # Calculate the maximum possible value of Oversegmentation loss.

        weight = self.weight if weight is None else weight
        N, C, H, W = seg_pred.shape
    
        # Calculate the number of nucleus and non-nucleus pixels
        total_pixels = N * H * W
        num_nucleus_pixels = torch.sum(batch_n[:, 0, :, :])
        num_non_nucleus_pixels = total_pixels - num_nucleus_pixels
    
        # Maximize count_cyto (all non-nucleus pixels predicted as cytoplasm)
        max_count_cyto = num_non_nucleus_pixels
    
        # Minimize count_nuc (all nucleus pixels predicted as non-nucleus)
        min_count_nuc = 0
    
        # Calculate the maximum extra overlap, then normalize and apply weight
        max_extra = max_count_cyto - min_count_nuc
        max_loss = (max_extra / N) * weight
    
        return max_loss.item()


class CellCallingLoss(nn.Module):
    """
    Maximise assignment of transcripts to cells
    """

    def __init__(self, weight, device) -> None:
        super(CellCallingLoss, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, seg_pred, batch_sa, weight=None):
        if weight is None:
            weight = self.weight
        
        # Limit to searchable area where there is detected expression
        penalisable = batch_sa * 1
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="none")
        loss = criterion_ce(seg_pred, penalisable[:, 0, :, :])

        loss_total = torch.sum(loss)

        loss_total = loss_total / seg_pred.shape[0]

        return weight * loss_total

    def get_max(self, seg_pred, batch_sa, weight=None):
        # Calculate the maximum possible value of CellCallingLoss.
        
        # Determine the number of classes (C) from seg_pred
        C = seg_pred.shape[1]
        N = seg_pred.shape[0]
    
        # Calculate the maximum CrossEntropy loss per pixel
        max_ce_loss = torch.log(torch.tensor(C, dtype=torch.float32))
    
        # Calculate the total number of searchable (penalisable) pixels
        total_penalisable_pixels = torch.sum(batch_sa[:, 0, :, :])
    
        # Compute the maximum total loss (averaged over batch size)
        max_loss = (max_ce_loss * total_penalisable_pixels) / N
    
        # Scale by the weight
        weight = self.weight if weight is None else weight
        max_loss = weight * max_loss.item()
    
        return max_loss


class OverlapLoss(nn.Module):
    """
    Penalise overlaps between different cells
    """

    def __init__(self, weight, device) -> None:
        super(OverlapLoss, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, seg_pred, batch_n, weight=None):
        if weight is None:
            weight = self.weight
        
        batch_n = batch_n[:, 0, :, :]
        seg_probs = torch.nn.functional.softmax(seg_pred, dim=1)

        all_nuclei = torch.sum(batch_n, 0)
        all_not_nuclei = torch.ones(batch_n.shape).to(self.device) - all_nuclei

        probs_cyto = seg_probs[:, 1, :, :] * all_not_nuclei

        ones = torch.ones(probs_cyto.shape).to(self.device)
        zeros = torch.zeros(probs_cyto.shape).to(self.device)

        # Penalise if number of cell prob > 0.5 is > 1
        alpha = 1.0
        preds_cyto = torch.sigmoid((probs_cyto - 0.5) * alpha) * (ones - zeros) + zeros
        count_cyto_overlap = torch.sum(preds_cyto, 0) - all_not_nuclei
        m = torch.nn.ReLU()
        count_cyto_overlap = m(count_cyto_overlap)

        loss = torch.sum(count_cyto_overlap)

        scale = seg_pred.shape[0] * seg_pred.shape[2] * seg_pred.shape[3]
        loss = loss / scale

        return weight * loss

    def get_max(self, seg_pred, batch_n, weight=None):
        # Calculate the maximum possible value of OverlapLoss.
        
        # Calculate the non-nuclei area
        all_nuclei = torch.sum(batch_n[:, 0, :, :], 0)
        all_not_nuclei = torch.ones_like(all_nuclei) - all_nuclei
    
        # Calculate the maximum count of cytoplasm overlap
        max_count_overlap = torch.sum(all_not_nuclei)
    
        # Normalize by the total number of pixels in the batch
        weight = self.weight if weight is None else weight
        N, C, H, W = seg_pred.shape
        max_loss = (max_count_overlap / (N * H * W)) * weight
    
        return max_loss.item()


class PosNegMarkerLoss(nn.Module):
    """
    Positive and negative markers of cell type
    """

    def __init__(self, weight_pos, weight_neg, device) -> None:
        super(PosNegMarkerLoss, self).__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.device = device

    def forward(self, seg_pred, batch_pos, batch_neg, weight_pos=None, weight_neg=None):
        if weight_pos is None:
            weight_pos = self.weight_pos
        if weight_neg is None:
            weight_neg = self.weight_neg
        
        batch_pos = batch_pos[:, 0, :, :]
        batch_neg = batch_neg[:, 0, :, :]

        # POSITIVE markers
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="sum")
        loss_pos = criterion_ce(seg_pred, batch_pos)

        # NEGATIVE markers
        seg_probs = torch.nn.functional.softmax(seg_pred, dim=1)
        probs_cells = seg_probs[:, 1, :, :]

        ones = torch.ones(probs_cells.shape).to(self.device)
        zeros = torch.zeros(probs_cells.shape).to(self.device)

        alpha = 1.0

        preds_cells = (
            torch.sigmoid((probs_cells - 0.5) * alpha) * (ones - zeros) + zeros
        )

        loss_neg = torch.sum(preds_cells * batch_neg)

        loss_total = (
            weight_pos * loss_pos + weight_neg * loss_neg
        ) / seg_pred.shape[0]

        return loss_total

    def get_max(self, seg_pred, batch_pos, batch_neg, weight_pos=None, weight_neg=None):
        # Calculate the maximum possible value of PosNegMarkerLoss.

        # Determine the number of classes (C) from seg_pred
        C = seg_pred.shape[1]
        N = seg_pred.shape[0]
    
        # Calculate the maximum CrossEntropy loss per positive pixel
        max_ce_loss = torch.log(torch.tensor(C, dtype=torch.float32))
    
        # Total number of positive and negative pixels
        total_pos_pixels = torch.sum(batch_pos[:, 0, :, :])
        total_neg_pixels = torch.sum(batch_neg[:, 0, :, :])
    
        # Calculate the maximum possible positive and negative losses
        weight_pos = self.weight_pos if weight_pos is None else weight_pos
        weight_neg = self.weight_neg if weight_neg is None else weight_neg
        max_loss_pos = weight_pos * max_ce_loss * total_pos_pixels
        max_loss_neg = weight_neg * total_neg_pixels
    
        # Normalize by the batch size
        max_loss = (max_loss_pos + max_loss_neg) / N
    
        return max_loss.item()
