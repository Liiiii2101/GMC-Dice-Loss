import torch
import torch.nn as nn

class SkelitePrecisionLoss(nn.Module):
    def __init__(self, skel_model, smooth=1., iter_=5):
        super(SkelitePrecisionLoss, self).__init__()
        self.skel_model = skel_model
        self.smooth = smooth
        self.skel_model.eval()  # only needed once

    def forward(self, y_pred, y_true, t_skeletonize_flage=False):
        # Get foreground probability
        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] 
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1]  # foreground prob

        with torch.no_grad():
            # Binarize GT and prediction
            y_true_bin = (y_true > 0).squeeze(1).float()
            y_pred_hard = (y_pred_prob > 0.5).float()

            skel_pred_hard, _ = self.skel_model(
                y_pred_hard.unsqueeze(1), z=None, no_iter=1, val_mode=True
            )## original iter is 5 but now it is set to 1 for roughness
            # skel_true, _ = self.skel_model(
            #     y_true_bin.unsqueeze(1), z=None, no_iter=5, val_mode=True
            # )

        skel_pred_hard = skel_pred_hard.squeeze(1)
        #skel_true = skel_true.squeeze(1)

        # Weighted by predicted probability
        skel_pred_prob = skel_pred_hard * y_pred_prob

        tprec = (torch.sum(skel_pred_prob * y_true_bin) + self.smooth) / \
                (torch.sum(skel_pred_prob) + self.smooth)
        # tsens = (torch.sum(skel_true * y_pred_prob) + self.smooth) / \
        #         (torch.sum(skel_true) + self.smooth)

        #cl_dice_loss = -2.0 * (tprec * tsens) / (tprec + tsens + 1e-8)

        return -tprec#cl_dice_loss
