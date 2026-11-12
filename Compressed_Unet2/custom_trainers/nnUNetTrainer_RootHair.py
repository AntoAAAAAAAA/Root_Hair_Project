
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


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