import numpy as np
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_and_loss.nnUNetTrainerUNETR import nnUNetTrainerUNETR
from nnunetv2.training.loss.compound_skelite_loss import DC_and_CE_and_Skelite_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from nnunetv2.Skelite.utils.model_utils import build_model, get_model, load_model_checkpoint
from nnunetv2.Skelite.data.utils import get_config, get_device

def load_skeleton_model():
    config_path = "/home/l.cai/small_bowel/small_bowel_data_code/skeleton-recall/nnunetv2/Skelite/pretrained/skelite_3d/config.yaml"
    checkpoint_path = "/home/l.cai/small_bowel/small_bowel_data_code/skeleton-recall/nnunetv2/Skelite/pretrained/skelite_3d/check/model_best.pt"
    config = get_config(config_path)
    device = get_device()
    model_module = get_model(config["net_type"])
    model = build_model(model_module, config, device)
    model = model.to(device)
    model = load_model_checkpoint(model, checkpoint_path, device)
    return model.eval()

# Load once
skel_model = load_skeleton_model()

class nnUNetTrainerUNETR_CE_DC_Skelite(nnUNetTrainerUNETR):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 200

    def _build_loss(self):
        if not hasattr(self, "skel_model"):
            config_path = "/home/l.cai/small_bowel/small_bowel_data_code/skeleton-recall/nnunetv2/Skelite/pretrained/skelite_3d/config.yaml"
            checkpoint_path = "/home/l.cai/small_bowel/small_bowel_data_code/skeleton-recall/nnunetv2/Skelite/pretrained/skelite_3d/check/model_best.pt"
            config = get_config(config_path)
            device = get_device()
            model_module = get_model(config["net_type"])
            self.skel_model = build_model(model_module, config, device)
            self.skel_model = self.skel_model.to(device)
            self.skel_model = load_model_checkpoint(self.skel_model, checkpoint_path, device)
            self.skel_model.eval()

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        # weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        # weights[-1] = 0

        # # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        # weights = weights / weights.sum()
        
        lambda_cldice = 1.0
        lambda_dice = 1.0
        lambda_ce = lambda_dice + lambda_cldice

        loss = DC_and_CE_and_Skelite_loss(self.skel_model, {'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                    {'iter_': 10, 'smooth': 1e-3},
                                    weight_ce=lambda_ce, weight_dice=lambda_dice, weight_cldice=lambda_cldice, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        self.print_to_log_file("lambda_cldice: %s" % str(lambda_cldice))
        self.print_to_log_file("lambda_dice: %s" % str(lambda_dice))
        self.print_to_log_file("lambda_ce: %s" % str(lambda_ce))

        # now wrap the loss
        #loss = DeepSupervisionWrapper(loss, weights)
        return loss
    