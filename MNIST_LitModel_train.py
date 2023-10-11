from Data.datamodules import MNISTDataModule
from Models.models import LitModel
import lightning as L

# Init DataModule
dm = MNISTDataModule(data_dir="./Data/MNIST")
# Init model from datamodule's attributes
model = LitModel(*dm.dims, dm.num_classes)
# Init trainer
trainer = L.Trainer(
    max_epochs=3,
    accelerator="auto",
    devices=1,
)
# Pass the datamodule as arg to trainer.fit to override model hooks :)
trainer.fit(model, dm)