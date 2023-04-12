import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

with open('avatars.html') as f:
    soup = BeautifulSoup(f.read(), 'html.parser').contents[0]

out_dir = 'data/avatars'
os.makedirs(out_dir, exist_ok=True)

for li in tqdm(soup):
    obj = li.contents[0].contents[0].contents[0]
    img_url = obj['src']
    label = obj['alt']
    img_url = img_url[:img_url.find('.png') + 4]
    r = requests.get(img_url)
    open(f'{out_dir}/{label}.png', 'wb').write(r.content)
