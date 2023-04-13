from functools import partial
import json
import os
import random
from pathlib import Path
from random import shuffle
import contextlib

import imgaug.augmenters as iaa
import cv2
import mlflow
import numpy as np
from torch import nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.ssd import SSD
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
import arrow
from torch.utils.data import DataLoader, Dataset
from ranger21 import Ranger21
from tqdm import tqdm
from torchvision.models.detection.ssdlite import SSDLiteHead, _normal_init, SSDLite320_MobileNet_V3_Large_Weights, _mobilenet_extractor
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class CircularDataset(Dataset):
    def __init__(self, samples, img_shape, aug=None):
        data = []
        self._aug = aug
        self._img_shape = img_shape
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
        with contextlib.suppress(TypeError):
            img = self._aug(image=img)
        zeros = np.zeros(self._img_shape + (3,))
        if self._aug:
            left = random.randint(0, self._img_shape[1] - img.shape[1])
            top = random.randint(0, self._img_shape[0] - img.shape[0])
        else:
            left = 0
            top = 0
        zeros[top:top + img.shape[0], left:left + img.shape[1]] = img
        boxes = [
            np.array(
                [
                    shape['points'][0][0] + left,
                    shape['points'][0][1] + top,
                    shape['points'][1][0] + left,
                    shape['points'][1][1] + top,
                ]
            )
            for shape in label
        ]
        boxes = np.array(boxes)
        return zeros.transpose((2, 0, 1)), boxes, np.ones((boxes.shape[0],))

def create_model(size, detections_per_img, topk_candidates, positive_fraction):
    weights = SSDLite320_MobileNet_V3_Large_Weights.verify(SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    weights_backbone = MobileNet_V3_Large_Weights.verify(MobileNet_V3_Large_Weights.DEFAULT)

    trainable_backbone_layers = _validate_trainable_layers(
        weights is not None or weights_backbone is not None, None, 6, 6
    )

    # Enable reduced tail if no pretrained backbone is selected. See Table 6 of MobileNetV3 paper.
    reduce_tail = weights_backbone is None

    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = mobilenet_v3_large(
        weights=weights_backbone, progress=True, norm_layer=norm_layer, reduced_tail=reduce_tail
    )
    if weights_backbone is None:
        # Change the default initialization scheme if not pretrained
        _normal_init(backbone)
    backbone = _mobilenet_extractor(
        backbone,
        trainable_backbone_layers,
        norm_layer,
    )

    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    if len(out_channels) != len(anchor_generator.aspect_ratios):
        raise ValueError(
            f"The length of the output channels from the backbone {len(out_channels)} do not match the length of the anchor generator aspect ratios {len(anchor_generator.aspect_ratios)}"
        )

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.9,
        "detections_per_img": detections_per_img,
        "topk_candidates": topk_candidates,
        "positive_fraction": positive_fraction,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, 1]
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    return SSD(
        backbone,
        anchor_generator,
        size,
        2,
        head=SSDLiteHead(out_channels, num_anchors, 2, norm_layer),
        **defaults,
    )


def main():
    os.makedirs(weight_dir, exist_ok=True)
    model = create_model(img_shape[::-1], detections_per_img, topk_candidates, positive_fraction).to(torch.float32).cuda()
    samples = [str(sample) for sample in Path(data_dir).glob('*.jpg')]
    train_count = round(0.8 * len(samples))
    shuffle(samples)
    train = samples[:train_count]
    val = samples[train_count:]
    train_loader = DataLoader(CircularDataset(
        train, img_shape, aug), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CircularDataset(val, img_shape), batch_size=batch_size)

    opt = Ranger21(model.parameters(), lr=lr, num_epochs=epochs, num_batches_per_epoch=len(train_loader),
                   warmdown_active=False, use_warmup=False)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)
    run_name = f"{arrow.now().format('MMDD_HHmm')}"
    weight_path = f'{weight_dir}/{run_name}.pth'
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        'lr': lr,
        'batch_size': batch_size,
        'topk_candidates': topk_candidates,
        'detections_per_img': detections_per_img,
        'positive_fraction': positive_fraction,
    })
    model.train()
    best_map = 0
    for epoch_i in tqdm(range(epochs)):
        losses = []
        for batch_x, boxes, labels in train_loader:
            targets = [
                {
                    'boxes': boxes[i].to(torch.long).cuda(),
                    'labels': labels[i].to(torch.long).cuda(),
                }
                for i in range(boxes.shape[0])
            ]
            opt.zero_grad(set_to_none=True)
            y_pred = model(batch_x.to(torch.float32).cuda(), targets)
            loss = y_pred['bbox_regression'] + y_pred['classification']
            loss.backward()
            opt.step()
            losses.append(loss.cpu().detach().numpy())
            scheduler.step()

        mlflow.log_metric('train_loss', np.mean(losses), epoch_i)
        if (epoch_i + 1) % eval_interval == 0:
            model.eval()
            val_map = MeanAveragePrecision()
            for batch_x, boxes, labels in val_loader:
                targets = [
                    {
                        'boxes': boxes[i].to(torch.long).cuda(),
                        'labels': labels[i].to(torch.long).cuda(),
                    }
                    for i in range(boxes.shape[0])
                ]
                with torch.no_grad():
                    y_pred = model(batch_x.cuda().to(torch.float32))
                val_map.update(y_pred, targets)
            val_map = val_map.compute()
            model.train()
            map_75 = val_map['map_75'].item()
            if map_75 > best_map:
                best_map = map_75
                torch.save(model.state_dict(), f'{weight_path[:-3]}best_map.pth')

            mlflow.log_metrics({
                'map': val_map['map'].item(),
                'map_50': val_map['map_50'].item(),
                'map_75': map_75,
            }, epoch_i)


if __name__ == '__main__':
    lr = 0.005
    positive_fraction = 0.7
    topk_candidates = 10
    detections_per_img = 2
    aug = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),
        iaa.Multiply((0.8, 1.2)),
        iaa.MultiplyHueAndSaturation((0.8, 1.2), per_channel=True),
        iaa.Cutout(nb_iterations=(1, 4), size=0.1, squared=False),
    ])
    epochs = 500
    eval_interval = 10
    batch_size = 32
    img_shape = (166, 762)
    weight_dir = 'weight/detector'
    data_dir = 'data/test_data/test_images'
    main()
