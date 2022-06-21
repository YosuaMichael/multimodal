import pytorch_lightning as pl
import torchmultimodal.models.omnivore as omnivore
import torch
from torch import nn
import utils


class OmnivoreLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = omnivore.omnivore_swin_t()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    def training_step(self, batch, batch_idx):
        (image, target), input_type = batch
        output = self.model(image, input_type)
        loss = self.criterion(output, target)
        self.log(f"train/losses/{input_type}", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (image, target), input_type = batch
        output = self.model(image, input_type)
        loss = self.criterion(output, target)
        self.log(f"val/losses/{input_type}", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        parameters = utils.set_weight_decay(
            self.model,
            self.args.weight_decay,
        )
        optimizer = torch.optim.SGD(
            parameters,
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        return optimizer



