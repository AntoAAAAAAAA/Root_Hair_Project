import numpy as np
import torch
import torch.nn as nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# from nnunetv2.utilities.helpers import set_default_inductor_cache_dir

# from custom_trainers.nnUNetTrainer_RootHair import nnUNetTrainer_RootHair
from custom_trainers.losses.cldice_loss import RootHairCLDiceLoss


class nnUNetTrainer_RootHair(nnUNetTrainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            device
        )

        self.num_epochs = 100
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 10

class RootHairCombinedLoss(nn.Module):
    """
    Standard nnU-Net Dice + CE:
        applied to all deep-supervision resolutions.

    Root-hair clDice:
        applied only to the highest-resolution output.

    Total:
        Dice+CE_deep_supervision + lambda * clDice_full_resolution
    """

    def __init__(self, base_loss, cldice_loss, cldice_weight=0.75):
        super().__init__()
        self.base_loss = base_loss
        self.cldice_loss = cldice_loss
        self.cldice_weight = cldice_weight

    def forward(self, net_output, target):
        base_loss_value = self.base_loss(net_output, target)

        full_resolution_output = net_output[0] if isinstance(net_output, (list, tuple)) else net_output
        full_resolution_target = target[0] if isinstance(target, (list, tuple)) else target

        cldice_loss_value = self.cldice_loss(
            full_resolution_output,
            full_resolution_target
        )

        return base_loss_value + self.cldice_weight * cldice_loss_value


class nnUNetTrainer_RootHair_CLDice(nnUNetTrainer_RootHair):    

    cldice_weight = 0.25
    cldice_iterations = 10

    def _build_loss(self):
        base_loss = DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )
        
        # This doesn't even work on mps, so no need 
        # if self._do_i_compile():
        #     set_default_inductor_cache_dir()
        #     base_loss.dc = torch.compile(base_loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            weights = np.array(
                [1 / (2 ** i) for i in range(len(deep_supervision_scales))]
            )

            weights[-1] = 1e-6 if self.is_ddp else 0
            weights /= weights.sum()

            base_loss = DeepSupervisionWrapper(base_loss, weights)

        cldice_loss = RootHairCLDiceLoss(
            root_hair_class=2,
            num_iter=self.cldice_iterations,
            smooth=1.0
        )

        return RootHairCombinedLoss(
            base_loss=base_loss,
            cldice_loss=cldice_loss,
            cldice_weight=self.cldice_weight
        )