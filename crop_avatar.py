import json
from pathlib import Path
import os

import cv2


def main():
    os.makedirs(out_dir, exist_ok=True)
    for i, sample in enumerate(Path(in_dir).glob('*.jpg')):
        sample = str(sample)
        json_path = f'{sample[:-3]}json'
        img = cv2.imread(sample)
        with open(json_path) as f:
            label = json.load(f)['shapes']
        for k, rec in enumerate(label):
            crop = img[round(rec['points'][0][1]):round(rec['points'][1][1]), round(rec['points'][0][0]):round(rec['points'][1][0])]
            cv2.imwrite(f'{out_dir}/{i}{k}.jpg', crop)

if __name__ == '__main__':
    in_dir = 'data/test_data/test_images'
    out_dir = 'data/crop'
    main()