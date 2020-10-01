import numpy as np
import matplotlib.pyplot as plt

from color import colorize
from noise import double_perlin3D

res = 2048
radius = 0.4

seed = 42

phi = np.linspace(0, 2 * np.pi, res)
theta = np.linspace(-np.pi / 2, np.pi / 2, res)

phi, theta = np.meshgrid(phi, theta)

x1 = radius * np.cos(theta) * np.cos(phi)
z1 = radius * np.cos(theta) * np.sin(phi)
y1 = radius * np.sin(theta)

x1 = x1.flatten()
y1 = y1.flatten()
z1 = z1.flatten()

h = double_perlin3D(x1, y1, z1, seed=seed)
vT = double_perlin3D(x1, y1, z1, a=20, persistance=0.8, seed=seed + 132)
vH = double_perlin3D(x1, y1, z1, a=20, persistance=0.8, seed=seed + 463)

rgbref = colorize(x1, y1, z1, h, vT, vH)

i_var = np.ones_like(h)

i_var[h > 0.5] = (0.5 + double_perlin3D(
    x1[h > 0.5],
    y1[h > 0.5],
    z1[h > 0.5],
    a=20,
    d=5,
    persistance=0.8,
    seed=seed + 214))

rgbref[:, 0] = rgbref[:, 0] * i_var
rgbref[:, 1] = rgbref[:, 1] * i_var
rgbref[:, 2] = rgbref[:, 2] * i_var
rgbref[:, 3] = rgbref[:, 3]

heightmap = np.zeros((res, res, 3))

heightmap[:, :, 0] = np.reshape(np.where(h > 0.5, (h - 0.5) * 2, 0),
                                (res, res))
heightmap[:, :, 1] = heightmap[:, :, 0]
heightmap[:, :, 2] = heightmap[:, :, 0]

rgb = np.zeros((res, res, 3))
rgb[:, :, 0] = np.reshape(rgbref[:, 0], (res, res))
rgb[:, :, 1] = np.reshape(rgbref[:, 1], (res, res))
rgb[:, :, 2] = np.reshape(rgbref[:, 2], (res, res))


spec = np.zeros((res, res, 3))
spec[:, :, 0] = np.reshape(rgbref[:, 3], (res, res))
spec[:, :, 1] = spec[:, :, 0]
spec[:, :, 2] = spec[:, :, 0]


plt.imsave("heightmap.png", heightmap)
plt.imsave("rgb.png", rgb.clip(0, 1))
plt.imsave("spec.png", spec)
