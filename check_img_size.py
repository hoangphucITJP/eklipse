from pathlib import Path

import cv2

max_h = None
max_w = None
for img_path in Path('data/test_data/test_images').glob('*.jpg'):
    img = cv2.imread(str(img_path))
    try:
        max_w = max(max_w, img.shape[1])
        max_h = max(max_h, img.shape[0])
    except TypeError:
        max_h = img.shape[0]
        max_w = img.shape[1]
print(max_h, max_w)
