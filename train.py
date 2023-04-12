import json
import random
from pathlib import Path
from random import shuffle

import cv2
import mlflow
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import arrow
from torch.utils.data import DataLoader, Dataset
from ranger21 import Ranger21
from tqdm import tqdm


class CircularDataset(Dataset):
    def __init__(self, samples):
        data = []
        for sample in samples:
            json_path = f'{sample[:-3]}json'
            with open(json_path) as f:
                label = json.load(f)['shapes']
            data.append((sample, label))
        self._samples = data

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, item):
        img_path, label = self._samples[item]
        img = cv2.imread(img_path)
        zeros = np.zeros(img_shape + (3,))
        left = random.randint(0, img_shape[1] - img.shape[1])
        top = random.randint(0, img_shape[0] - img.shape[0])
        zeros[top:top + img.shape[0], left:left + img.shape[1]] = img

        boxes = []
        for shape in label:
            boxes.append(
                np.array([shape['points'][0][0] + left, shape['points'][0][1] + top, shape['points'][1][0] + left,
                          shape['points'][1][1] + top]))
        boxes = np.array(boxes)
        return zeros.transpose((2, 0, 1)), boxes, np.zeros((boxes.shape[0],))


def main():
    model = ssdlite320_mobilenet_v3_large(num_classes=1).to(torch.float32)
    samples = [str(sample) for sample in Path(data_dir).glob('*.jpg')]
    train_count = round(0.8 * len(samples))
    shuffle(samples)
    train = samples[:train_count]
    val = samples[train_count:]
    train_loader = DataLoader(CircularDataset(train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CircularDataset(val), batch_size=batch_size)

    opt = Ranger21(model.parameters(), lr=lr, num_epochs=epochs, num_batches_per_epoch=len(train_loader),
                   warmdown_active=False, use_warmup=False)
    swa_model = AveragedModel(model)
    swa_start = int(epochs * 75 / 100)
    swa_scheduler = SWALR(opt, swa_lr=lr, anneal_epochs=swa_start, anneal_strategy='linear')
    scheduler = CosineAnnealingLR(opt, T_max=epochs)
    mlflow.start_run(run_name=f"{arrow.now().format('MM-DD HH:mm')}")
    for epoch_i in tqdm(range(epochs)):
        model.train()
        losses = []
        for batch_x, boxes, labels in train_loader:
            targets = []
            for i in range(boxes.shape[0]):
                targets.append({
                    'boxes': boxes[i].to(torch.long),
                    'labels': labels[i].to(torch.long),
                })
            opt.zero_grad(set_to_none=True)
            y_pred = model(batch_x.to(torch.float32), targets)
            loss = y_pred['bbox_regression']
            loss.backward()
            opt.step()
            losses.append(loss.detach().numpy())

            if epoch_i > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

        mlflow.log_metric('train_loss', np.mean(losses), epoch_i)
        if (epoch_i + 1 % eval_interval) == 0:
            model.eval()
            losses = []
            for batch_x, boxes, labels in val_loader:
                targets = []
                for i in range(boxes.shape[0]):
                    targets.append({
                        'boxes': boxes[i].to(torch.long),
                        'labels': labels[i].to(torch.long),
                    })
                with torch.no_grad:
                    y_pred = model(batch_x.to(torch.float32), targets)
                losses.append(y_pred['bbox_regression'].numpy())
                mlflow.log_metric('val_loss', np.mean(losses), epoch_i)

    torch.optim.swa_utils.update_bn(train_loader, swa_model)


if __name__ == '__main__':
    lr = 0.001
    epochs = 100
    eval_interval = 5
    batch_size = 4
    img_shape = (166, 762)
    data_dir = 'data/test_data/test_images'
    main()
