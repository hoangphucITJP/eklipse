import os
from pathlib import Path

import torch
from tqdm import tqdm
import cv2

from train_re_id import ft_net



def main():
    os.makedirs(out_dir, exist_ok=True)
    model = ft_net(num_classes, return_f=True).float().cuda()
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    for img_path in tqdm(Path(data_dir).rglob('*.*')):
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, img_shape)

        # -------- forward --------
        with torch.no_grad():
            _, feature = model(torch.from_numpy(img.transpose((2, 0, 1))).float().cuda().unsqueeze(0))
        ff = feature.data.cpu()
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        torch.save(ff, f'{out_dir}/{img_path.name[:-3]}feat')


if __name__ == '__main__':
    num_classes = 51
    img_shape = (256, 256)
    weight_path = 'weight/re_id/0413_0241.pth'
    data_dir = 'data/avatars'
    out_dir = 'data/features'
    main()
