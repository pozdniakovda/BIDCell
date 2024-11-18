import torch
import torch.nn as nn


class NucEncapOverlapLoss(nn.Module):
    """
    Combines NucleiEncapsulationLoss and OverlapLoss.
    This loss ensures that nuclei are encapsulated within predicted cells and penalizes overlaps between different cells.
    """

    def __init__(self, weight_encap, weight_overlap, device) -> None:
        super(NucEncapOverlapLoss, self).__init__()
        self.nuclei_encapsulation_loss = NucleiEncapsulationLoss(weight_encap, device)
        self.overlap_loss = OverlapLoss(weight_overlap, device)

    def forward(self, seg_pred, batch_n, weight_encap=None, weight_overlap=None):
        # Compute NucleiEncapsulationLoss
        encap_loss = self.nuclei_encapsulation_loss(
            seg_pred, batch_n, weight=weight_encap
        )

        # Compute OverlapLoss
        overlap_loss = self.overlap_loss(
            seg_pred, batch_n, weight=weight_overlap
        )

        # Return the sum of both losses
        return encap_loss + overlap_loss

    def get_max(self, input_shape, weight_encap=None, weight_overlap=None):
        # Compute maximum possible NucleiEncapsulationLoss
        max_encap_loss = self.nuclei_encapsulation_loss.get_max(
            input_shape, weight=weight_encap
        )

        # Compute maximum possible OverlapLoss
        max_overlap_loss = self.overlap_loss.get_max(
            input_shape, weight=weight_overlap
        )

        # Return the sum of both maximum losses
        return max_encap_loss + max_overlap_loss


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

    def get_max(self, input_shape, cc_weight=None, weight_pos=None, weight_neg=None):
        # Compute maximum possible CellCallingLoss
        max_calling_loss = self.cell_calling_loss.get_max(
            input_shape, weight=cc_weight
        )

        # Compute maximum possible PosNegMarkerLoss
        max_marker_loss = self.pos_neg_marker_loss.get_max(
            input_shape, weight_pos=weight_pos, weight_neg=weight_neg
        )

        # Return the sum of both maximum losses
        return max_calling_loss + max_marker_loss


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

        # This loss is simply a CrossEntropyLoss
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion_ce(seg_pred, batch_n[:, 0, :, :])

        return self.weight * loss

    def get_max(self, input_shape, weight=None):
        '''
        CrossEntropyLoss is maximized when predictions are completely wrong.
        The worst-case scenario assumes that for every pixel, the predicted class has the lowest probability 
        (close to zero), and the true class is completely misclassified.
        '''
        
        batch_size, num_classes, height, width = input_shape
    
        # The maximum loss per pixel is given by -log(1/num_classes).
        max_loss_per_pixel = -torch.log(torch.tensor(1.0 / num_classes)).to(self.device)
    
        # Compute the total maximum loss for all pixels in the batch
        total_max_loss = max_loss_per_pixel * batch_size * height * width

        # Apply the weight
        weight = self.weight if weight is None else weight
        max_loss = weight * max_loss

        return max_loss.item()


class OversegmentationLoss(nn.Module):
    """
    Minimise oversegmentation
    """

    def __init__(self, weight, device) -> None:
        super(OversegmentationLoss, self).__init__()
        self.weight = weight
        self.init_weight = weight
        self.device = device
    
    def forward(self, seg_pred, batch_n, weight=None):
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
        mask_cyto = 1.0 - batch_n
        probs_cyto = seg_probs[:, 1, :, :] * mask_cyto
    
        # Apply sigmoid to emphasize probabilities > 0.5
        preds_nuc = torch.sigmoid((probs_nuc - 0.5))
        count_nuc = torch.sum(preds_nuc)
    
        preds_cyto = torch.sigmoid((probs_cyto - 0.5))
        count_cyto = torch.sum(preds_cyto)
    
        # Compute extra cytoplasm predictions (oversegmentation)
        extra = torch.nn.ReLU()(count_cyto - count_nuc)
    
        # Normalize by batch size and apply weight
        loss = (extra / seg_pred.shape[0]) * self.weight
    
        return loss

    def get_max(self, input_shape, weight=None):
        '''
        In the worst-case scenario, the entire image is predicted as cytoplasm,
        and there are no nuclei predictions, which maximizes the count difference.
        '''
        
        batch_size, num_classes, height, width = input_shape

        max_count_cyto = batch_size * height * width  # All pixels are cytoplasm
        max_count_nuc = 0  # No pixels are classified as nuclei
    
        # The maximum extra count is the difference: max_count_cyto - max_count_nuc
        max_extra = max_count_cyto - max_count_nuc
    
        # Since ReLU is applied, max_extra is already non-negative.
        max_loss = max_extra / batch_size # normalize by the batch size (as in the forward method)

        # Apply the weight
        weight = self.weight if weight is None else weight
        max_loss = weight * max_loss
        
        return max_loss.item()


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

        loss_total = loss_total / seg_pred.shape[0]

        return self.weight * loss_total

    def get_max(self, input_shape, weight=None):
        '''
        In the worst-case scenario, every pixel within the searchable area
        is misclassified, maximizing the CrossEntropy loss.
        '''
        
        batch_size, num_classes, height, width = input_shape
    
        # The maximum loss per pixel occurs when the predicted probability for the true class is zero.
        # This corresponds to a loss of -log(1/num_classes) for each pixel.
        max_loss_per_pixel = -torch.log(torch.tensor(1.0 / num_classes)).to(self.device)
    
        # Total maximum loss across all pixels in the batch
        max_loss_total = max_loss_per_pixel * batch_size * height * width
    
        # Normalize by the batch size (as in the forward method)
        max_loss = max_loss_total / batch_size
    
        # Apply the weight
        weight = self.weight if weight is None else weight
        max_loss = weight * max_loss
        
        return max_loss.item()


class OverlapLoss(nn.Module):
    """
    Penalise overlaps between different cells
    """

    def __init__(self, weight, device) -> None:
        super(OverlapLoss, self).__init__()
        self.weight = weight
        self.init_weight = weight
        self.device = device

    def forward(self, seg_pred, batch_n, weight=None):
        # Overwrite self.weight if new weight is given; original is preserved as self.init_weight
        if weight is not None:
            self.weight = weight
        
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

        return self.weight * loss
    
    def get_max(self, input_shape, weight=None):
        '''
        In the worst-case scenario, every pixel is predicted as cytoplasm with high overlap,
        resulting in the maximum possible overlap penalty.
        '''
        
        batch_size, num_classes, height, width = input_shape
    
        # In the worst case, every pixel is classified as cytoplasm with maximum probability.
        max_count_cyto_overlap = batch_size * height * width  # All pixels contribute to overlap
    
        # Normalization factor based on the total number of pixels
        scale = batch_size * height * width
        max_loss = max_count_cyto_overlap / scale
    
        # Apply the weight
        weight = self.weight if weight is None else weight
        max_loss = weight * max_loss
    
        return max_loss.item()


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

    def forward(self, seg_pred, batch_pos, batch_neg, weight_pos=None, weight_neg=None):
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

    def get_max(self, input_shape, weight_pos=None, weight_neg=None):
        '''
        In the worst-case scenario:
        - For positive markers, every pixel is misclassified, maximizing CrossEntropy loss.
        - For negative markers, every pixel is predicted as a cell, maximizing the overlap with the negative mask.
        '''
    
        batch_size, num_classes, height, width = input_shape
    
        # Set default weights if not provided
        weight_pos = self.weight_pos if weight_pos is None else weight_pos
        weight_neg = self.weight_neg if weight_neg is None else weight_neg
    
        # MAXIMUM POSITIVE LOSS:
        # Assume every pixel is misclassified for the positive markers (worst case).
        max_loss_pos_per_pixel = -torch.log(torch.tensor(1.0 / num_classes)).to(self.device)
        max_loss_pos = max_loss_pos_per_pixel * batch_size * height * width
    
        # MAXIMUM NEGATIVE LOSS:
        # Assume every pixel is classified as a cell, maximizing overlap with the negative mask.
        max_loss_neg = batch_size * height * width  # All pixels contribute to the negative loss.
    
        # Normalize losses by the batch size
        max_loss_pos = max_loss_pos / batch_size
        max_loss_neg = max_loss_neg / batch_size
    
        # Apply weights
        max_loss = (weight_pos * max_loss_pos) + (weight_neg * max_loss_neg)
    
        return max_loss.item()
