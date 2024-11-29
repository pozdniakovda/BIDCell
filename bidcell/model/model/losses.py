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
        self.device = device

    def forward(self, seg_pred, batch_n):
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion_ce(seg_pred, batch_n[:, 0, :, :])

        return self.weight * loss


class Oversegmentation(nn.Module):
    """
    Minimise oversegmentation
    """

    def __init__(self, weight, device) -> None:
        super(Oversegmentation, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, seg_pred, batch_n):
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

        return self.weight * loss


class CellCallingLoss(nn.Module):
    """
    Maximise assignment of transcripts to cells
    """

    def __init__(self, weight, device) -> None:
        super(CellCallingLoss, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, seg_pred, batch_sa):
        # Limit to searchable area where there is detected expression
        penalisable = batch_sa * 1
        criterion_ce = torch.nn.CrossEntropyLoss(reduction="none")
        loss = criterion_ce(seg_pred, penalisable[:, 0, :, :])

        loss_total = torch.sum(loss)

        loss_total = loss_total / seg_pred.shape[0]

        return self.weight * loss_total


class OverlapLoss(nn.Module):
    """
    Penalise overlaps between different cells
    """

    def __init__(self, weight, device) -> None:
        super(OverlapLoss, self).__init__()
        self.weight = weight
        self.device = device

    def forward(self, seg_pred, batch_n):
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

    def get_max(self, input_shape, num_transcripts, weight=None):
        """
        Compute the maximum possible loss assuming the worst-case scenario.
        
        Args:
            input_shape: Shape of seg_pred (batch_size, num_classes, height, width).
            num_transcripts: Number of transcripts in each image (tensor of shape [batch_size]).
            weight: Optional weight to override the initialized weight.

        Returns:
            float: Maximum possible loss value.
        """
        batch_size, _, height, width = input_shape
        max_loss = 0.0

        for b in range(batch_size):
            if num_transcripts[b] > 0:
                # Worst case: Each transcript is assigned fully to multiple cells
                penalty = num_transcripts[b].item()  # Each transcript contributes a maximum penalty of 1
            else:
                penalty = 0.0  # No transcripts, no penalty

            max_loss += penalty

        max_loss = (max_loss / batch_size) * (self.weight if weight is None else weight)
        return max_loss


class PosNegMarkerLoss(nn.Module):
    """
    Positive and negative markers of cell type
    """

    def __init__(self, weight_pos, weight_neg, device) -> None:
        super(PosNegMarkerLoss, self).__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
        self.device = device

    def forward(self, seg_pred, batch_pos, batch_neg):
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
            self.weight_pos * loss_pos + self.weight_neg * loss_neg
        ) / seg_pred.shape[0]

        return loss_total
