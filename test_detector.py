from pathlib import Path
import os

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import save_image

from train_detector import CircularDataset, create_model


def main():
    vis_dir = f'vis/{Path(weight_path).name[:-4]}'
    os.makedirs(vis_dir, exist_ok=True)
    model = create_model(img_shape[::-1], detections_per_img, topk_candidates, positive_fraction).to(torch.float32).cuda()
    samples = [str(sample) for sample in Path(data_dir).glob('*.jpg')]
    data_loader = DataLoader(CircularDataset(samples, img_shape))

    model.load_state_dict(torch.load(weight_path))
    model.eval()
    map_metric = MeanAveragePrecision()
    for k, (batch_x, boxes, labels) in tqdm(enumerate(data_loader)):
        targets = [
            {
                'boxes': boxes[i].to(torch.long).cuda(),
                'labels': labels[i].to(torch.long).cuda(),
            }
            for i in range(boxes.shape[0])
        ]
        with torch.no_grad():
            y_pred = model(batch_x.cuda().to(torch.float32))

        save_image(draw_bounding_boxes(batch_x[0].to(torch.uint8), y_pred[0]['boxes']) / 255, f'{vis_dir}/{k}.jpg')
        map_metric.update(y_pred, targets)
    map_metric = map_metric.compute()
    print(f'mAP: {map_metric["map"].item()}\n')
    print(f'mAP75: {map_metric["map_75"].item()}\n')
    print(f'mAP50: {map_metric["map_50"].item()}\n')


if __name__ == '__main__':
    positive_fraction = 0.7
    topk_candidates = 10
    detections_per_img = 2
    img_shape = (166, 762)
    weight_path = 'weight/detector/0412_2334.best_map.pth'
    data_dir = 'data/test_data/test_images'
    main()
