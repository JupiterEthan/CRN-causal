import torch


class LossFunction(object):
    def __call__(self, est, lbl, loss_mask, n_frames):
        est_t = est * loss_mask
        lbl_t = lbl * loss_mask

        n_feats = est.shape[-1]

        loss = torch.sum((est_t - lbl_t)**2) / float(sum(n_frames) * n_feats)
        
        return loss
