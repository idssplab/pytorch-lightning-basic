from Data.datamodules import CIFAR10DataModule
from Models.models import LitResnet
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

# Init DataModule
dm = CIFAR10DataModule(data_dir="./Data/CIFAR10")
# Init model from datamodule's attributes
model = LitResnet(dm.dims[0], dm.num_classes)

ckpt_cb = ModelCheckpoint(filename='{epoch:d}',
                          monitor='val_acc',
                          mode='max',
                          save_top_k=5)
pbar = TQDMProgressBar(refresh_rate=1)
callbacks = [ckpt_cb, pbar]

logger=CSVLogger(save_dir="logs/",
                 name='CIFAR10_LitResnet_test')

# Init trainer
trainer = L.Trainer(
    callbacks=callbacks,
    logger=logger,
    max_epochs=10,
    accelerator="auto",
    devices=1,
)
# Pass the datamodule as arg to trainer.fit to override model hooks :)
trainer.fit(model, dm)