import numpy as np
import matplotlib.pyplot as plt

import os

from tqdm import tqdm
from main import gen_rgbbase_and_normal, apply_illum
import shutil

fig = plt.figure()

N_seconds = 20
FPS = 20

N_frame = FPS * N_seconds

phi = np.linspace(0, 2 * np.pi, N_frame)

ims = []

os.mkdir('temp')

for i in tqdm(range(N_frame)):
    rgbref_base, normal_base, cmap = gen_rgbbase_and_normal(seed=436, rot=phi[i])
    im = apply_illum(rgbref_base, normal_base, cmap)
    plt.imsave("temp/out" + str(i) + ".png", im)

os.system(
    "ffmpeg -r " + str(FPS) +
    " -f image2 -s 1024x1024 -i temp/out%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4"
)

shutil.rmtree('temp')