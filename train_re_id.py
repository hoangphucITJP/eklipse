import contextlib
import os
from pathlib import Path
from random import shuffle

import cv2
import imgaug.augmenters as iaa
import mlflow
import numpy as np
from torch import nn
import torch
from torch.nn import init
from torch.optim.lr_scheduler import CosineAnnealingLR
import arrow
from torchmetrics import F1Score
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
from ranger21 import Ranger21
from tqdm import tqdm


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # For old pytorch, you may use kaiming_normal.
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear > 0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


class ReIdDataset(Dataset):
    def __init__(self, samples, img_shape, aug=None):
        self._aug = aug
        self._img_shape = img_shape
        data = [(sample, int(Path(sample).parts[-2])) for sample in samples]
        self._samples = data

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, item):
        img_path, label = self._samples[item]
        img = cv2.imread(img_path)
        with contextlib.suppress(TypeError):
            img = self._aug(image=img)
        img = cv2.resize(img, self._img_shape)
        return img.transpose((2, 0, 1)), label


class ft_net(nn.Module):
    def __init__(self, class_num, drop_rate=0., return_f=False):
        super(ft_net, self).__init__()
        # load the model
        model_ft = resnet50(pretrained=True)
        # change avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(1024, class_num, droprate=drop_rate, return_f=return_f)  # define our classifier.

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.avgpool(x)
        x = self.classifier(x[..., 0, 0])  # use our classifier.
        return x


def main():
    os.makedirs(weight_dir, exist_ok=True)
    samples = [str(sample) for sample in Path(data_dir).rglob('*.*')]
    train_count = round(0.8 * len(samples))
    shuffle(samples)
    train = samples[:train_count]
    val = samples[train_count:]
    train_loader = DataLoader(ReIdDataset(
        train, img_shape, aug), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ReIdDataset(val, img_shape), batch_size=batch_size)
    model = ft_net(num_classes, drop_rate).float().cuda()
    opt = Ranger21(model.parameters(), lr=lr, num_epochs=epochs, num_batches_per_epoch=len(train_loader),
                   warmdown_active=False, use_warmup=False)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)
    run_name = f"{arrow.now().format('MMDD_HHmm')}"
    weight_path = f'{weight_dir}/{run_name}.pth'
    mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        'lr': lr,
        'batch_size': batch_size,
        'drop_rate': drop_rate,
    })
    model.train()
    criterion = nn.CrossEntropyLoss()
    f1 = F1Score(task="multiclass", num_classes=num_classes)
    best_f1 = 0
    for epoch_i in tqdm(range(epochs)):
        losses = []

        preds = []
        targets = []
        for data in train_loader:
            # get a batch of inputs
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad(set_to_none=True)

            # -------- forward --------
            outputs = model(inputs.float().cuda())
            _, _ = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels.cuda())

            # -------- backward + optimize --------
            # only if in training phase
            loss.backward()
            opt.step()
            losses.append(loss.cpu().detach().numpy())
            scheduler.step()
            preds.append(outputs.cpu())
            targets.append(labels)

        mlflow.log_metrics({
            'train_f1': f1(torch.cat(preds), torch.cat(targets)).item(),
            'train_loss': np.mean(losses),
        }, epoch_i)
        if (epoch_i + 1) % eval_interval == 0:
            model.eval()
            preds = []
            targets = []
            losses = []
            for data in val_loader:
                # get a batch of inputs
                inputs, labels = data

                # -------- forward --------
                with torch.no_grad():
                    outputs = model(inputs.float().cuda())
                _, _ = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels.cuda())

                # -------- backward + optimize --------
                # only if in training phase
                losses.append(loss.cpu().detach().numpy())

                preds.append(outputs.cpu())
                targets.append(labels)
            val_f1 = f1(torch.cat(preds), torch.cat(targets))
            if val_f1 > best_f1:
                torch.save(model.state_dict(), weight_path)
                best_f1 = val_f1
            mlflow.log_metrics({
                'val_f1': val_f1.item(),
                'val_loss': np.mean(losses),
            }, epoch_i)
            model.train()


if __name__ == '__main__':
    lr = 0.0025
    aug = iaa.Sequential([
        iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}),
        iaa.AdditiveGaussianNoise(scale=(0, 0.01*255)),
        iaa.Multiply((0.8, 1.2)),
        iaa.MultiplyHueAndSaturation((0.8, 1.2), per_channel=True),
        iaa.Cutout(nb_iterations=(1, 7), size=0.2, squared=False),
    ])
    epochs = 500
    eval_interval = 10
    batch_size = 16
    drop_rate = 0.3
    num_classes = 51
    img_shape = (256, 256)
    weight_dir = 'weight/re_id'
    data_dir = 'data/re_id'
    main()
