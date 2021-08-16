import torch
import torch.nn as nn
import torch.nn.functional as F

class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def forward(self, logits, true, eps=1e-7):
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = torch.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))

        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)


class jacc_loss(nn.Module):
    def __init__(self):
        super(jacc_loss, self).__init__()

    def forward(self, logits, true, eps=1e-7):
        """Computes the Jaccard loss, a.k.a the IoU loss.
            Note that PyTorch optimizers minimize a loss. In this
            case, we would like to maximize the jaccard loss so we
            return the negated jaccard loss.
            Args:
                true: a tensor of shape [B, H, W] or [B, 1, H, W].
                logits: a tensor of shape [B, C, H, W]. Corresponds to
                    the raw output or logits of the model.
                eps: added to the denominator for numerical stability.
            Returns:
                jacc_loss: the Jaccard loss.
            """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = torch.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + eps)).mean()
        return (1 - jacc_loss)


class Instance_Aware_Loss(nn.Module):
    def __init__(self, eps=1e-5):
        super(Instance_Aware_Loss, self).__init__()
        self.eps = eps

    def forward(self, logits, true,weight):

        num_classes = logits.shape[1]

        true_dummy = torch.eye(num_classes)[true.squeeze(1)]
        true_dummy = true_dummy.permute(0, 3, 1, 2)
        probas = F.softmax(logits, dim=1)
        true_dummy = true_dummy.type(logits.type())

        probas=probas[:,1:2,...]
        true_dummy=true_dummy[:,1:2,...]

        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_dummy*weight, dims)
        cardinality = torch.sum(probas*weight + true_dummy*weight, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)

class GeneralizedDiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(GeneralizedDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, true):
        """
        Implementation of generalized dice loss for multi-class semantic segmentation
        Args:
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        true: a tensor of shape [B, 1, H, W].
        Returns:
        dice_loss: the SÃ¸rensenâ€“Dice loss.
        """
        num_classes = logits.shape[1]

        true_dummy = torch.eye(num_classes)[true.squeeze(1)]
        true_dummy = true_dummy.permute(0, 3, 1, 2)
        probas = F.softmax(logits, dim=1)

        true_dummy = true_dummy.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_dummy, dims)
        cardinality = torch.sum(probas + true_dummy, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return 1 - dice_loss

