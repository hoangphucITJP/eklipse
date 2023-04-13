import argparse
from pathlib import Path
import os

import torch
from tqdm import tqdm
import cv2
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import save_image
from torchvision.transforms import Resize
import numpy as np
from torch import nn

from train_detector import create_model
from train_re_id import ft_net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    
    vis_dir = f'vis/{Path(weight_path).name[:-4]}'
    os.makedirs(vis_dir, exist_ok=True)
    model = create_model(img_shape[::-1], detections_per_img, topk_candidates, positive_fraction).to(torch.float32).cuda()

    model.load_state_dict(torch.load(weight_path))
    model.eval()

    re_id = ft_net(num_classes, return_f=True).float().cuda()
    re_id.load_state_dict(torch.load(re_id_weight))
    re_id.eval()
    avatar_resize = Resize(avatar_size)
    gallery = [
        (torch.load(feat_path), feat_path.name[:-5])
        for feat_path in Path(gallery_dir).glob('*.feat')
    ]
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    with open(args.output, 'w') as f:
        for k, img_path in tqdm(enumerate(Path(args.input).glob('*.jpg'))):
            img = cv2.imread(str(img_path))
            zeros = np.zeros(img_shape + (3,))
            zeros[:img.shape[0], :img.shape[1]] = img
            tens = torch.from_numpy(zeros.transpose((2, 0, 1)))
            with torch.no_grad():
                y_pred = model([tens.cuda().to(torch.float32)])

            boxes = y_pred[0]['boxes']
            idx = boxes[:, 0].argmin().item()
            box = boxes[[idx]]
            crop = tens[:, round(box[0][1].item()):round(box[0][3].item()),round(box[0][0].item()):round(box[0][2].item())]
            sized = avatar_resize(crop)

            with torch.no_grad():
                _, feature = re_id(sized.float().cuda().unsqueeze(0))
            ff = feature.data.cpu()
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            idx = np.array([cos(ff[0], cham[0][0]) for cham in gallery]).argmax()
            label = gallery[idx][1]

            save_image((draw_bounding_boxes(tens[[2, 1, 0]].to(torch.uint8), box, [label], colors=[(0, 0, 255)]) / 255), f'{vis_dir}/{k}.jpg')
            f.write(f'{img_path.name}\t{label}\n')

if __name__ == '__main__':
    positive_fraction = 0.7
    topk_candidates = 10
    gallery_dir = 'data/features'
    detections_per_img = 2
    img_shape = (166, 762)
    avatar_size = (256, 256)
    num_classes = 51
    weight_path = 'weight/detector/0412_2334.best_map.pth'
    re_id_weight = 'weight/re_id/0413_0241.pth'
    main()
