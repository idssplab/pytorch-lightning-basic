from Data.datamodules import CIFAR10DataModule
from Models.models import LitModel
import lightning as L

# Init DataModule
dm = CIFAR10DataModule(data_dir="./Data/CIFAR10")
# Init model from datamodule's attributes
model = LitModel(*dm.dims, dm.num_classes, hidden_size=256)
# Init trainer
trainer = L.Trainer(
    max_epochs=30,
    accelerator="auto",
    devices=1,
)
# Pass the datamodule as arg to trainer.fit to override model hooks :)
trainer.fit(model, dm)