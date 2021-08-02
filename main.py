import torch
import timm

import numpy as np
import PIL.Image as Image
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from src.util.sam import SAM
from pytorch_lightning import LightningModule, Trainer

bs = 2


class GravWaveDataset(Dataset):

    def __init__(self, data, data_dir):
        self.data = data
        self.data_dir = data_dir

        self.images = [np.array(Image.open("{}/{}/{}.jpg".format(data_dir, target, id_))).transpose((2, 0, 1))
                       for id_, target in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.images[index].astype(np.float32) / 255

        img = torch.tensor(img)

        # print(img.shape)

        return img, torch.tensor(self.data[index][1], dtype=torch.float32)

data_dir = "./data"

train_data = np.load("./data/train_info.npy", allow_pickle=True)
val_data = np.load("./data/validation_info.npy", allow_pickle=True)

train_ds = GravWaveDataset(train_data, "./data/train")
val_ds = GravWaveDataset(val_data, "./data/validation")

train_dl = DataLoader(train_ds, shuffle=True, batch_size=bs)
val_dl = DataLoader(val_ds, batch_size=bs)


class GravModel(LightningModule):

    def __init__(self, useSAM=True):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_l2_ns_475', pretrained=True, num_classes=1)

        self.useSAM = useSAM

        if self.useSAM:
            self.automatic_optimization = False

    def configure_optimizers(self):
        if self.useSAM:
            base_optimizer = torch.optim.SGD
            optimizer = SAM(self.parameters(), base_optimizer, lr=0.1)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        #         optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        return optimizer

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.useSAM:
            optimizer = self.optimizers()

            optimizer.zero_grad()
            # first forward-backward pass
            loss_1 = self.compute_loss(x, y)
            self.manual_backward(loss_1)
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            loss_2 = self.compute_loss(x, y)
            self.manual_backward(loss_2)
            optimizer.second_step(zero_grad=True)

            return loss_1

        else:

            loss = self.compute_loss(x, y)

    def validation_step(self, batch, batch_idx):
        # print(batch)
        x, y = batch

        with torch.no_grad():
            loss = self.compute_loss(x, y)

        return loss

    def compute_loss(self, x, y):
        logits = self.forward(x)
        return F.mse_loss(logits, y)

module = GravModel()

trainer = Trainer(fast_dev_run=True, num_sanity_val_steps=1)

trainer.fit(module, train_dl, val_dl)