import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from noise import double_perlin3D, background

from geometry import normal, cos_vec, reflection
from color import colorize

res = 1024
radius = 0.4

multi_sampling = 1

eps = 1 / (100 * res)

atmos_size = 0.02
atmos_color = (0.6, 0.851, 0.912)

x, y = np.meshgrid(np.linspace(-0.5, 0.5, res), np.linspace(-0.5, 0.5, res))

dx, dy = np.meshgrid(
    np.linspace(0, 1 / res, multi_sampling),
    np.linspace(0, 1 / res, multi_sampling))


def gen_rgbbase_and_normal(seed=69, rot=0):
    im_rgbref_base = np.zeros((res, res, multi_sampling, multi_sampling, 4))
    normal_base = np.zeros((res, res, multi_sampling, multi_sampling, 3))
    cloud_map = np.zeros((res, res, multi_sampling, multi_sampling))

    for i in tqdm(range(res)):
        for d1 in range(multi_sampling):
            for d2 in range(multi_sampling):

                z2 = radius**2 - (x[i, :] + dx[d1, d2])**2 - (
                    y[i, :] + dy[d1, d2])**2
                ok = z2 >= 0

                if np.any(ok):
                    x1, y1, z1 = (x[i, ok] + dx[d1, d2]), (
                        y[i, ok] + dy[d1, d2]), np.sqrt(z2[ok])

                    ox1, oz1 = x1, z1

                    x1, z1, = x1 * np.cos(rot) - z1 * np.sin(rot), x1 * np.sin(
                        rot) + z1 * np.cos(rot)

                    h = double_perlin3D(x1, y1, z1, seed=seed)

                    vT = double_perlin3D(
                        x1, y1, z1, a=20, persistance=0.8, seed=seed + 132)
                    vH = double_perlin3D(
                        x1, y1, z1, a=20, persistance=0.8, seed=seed + 463)

                    c = double_perlin3D(
                        x1, y1, z1, a=5, d=5, persistance=0.6, seed=seed + 212)

                    c = .6 * c + .4 * double_perlin3D(
                        x1,
                        y1,
                        z1,
                        a=20,
                        d=10,
                        persistance=0.8,
                        seed=seed + 213)

                    cloud_map[i, ok, d1, d2] = c

                    rgbref = colorize(x1, y1, z1, h, vT, vH)

                    dhdx = np.zeros_like(h)
                    dhdy = np.zeros_like(h)
                    dhdz = np.zeros_like(h)

                    dhdx[h > 0.5] = (double_perlin3D(
                        x1[h > 0.5] + eps, y1[h > 0.5], z1[h > 0.5], seed=seed)
                                     - h[h > 0.5]) / eps
                    dhdy[h > 0.5] = (double_perlin3D(
                        x1[h > 0.5], y1[h > 0.5] + eps, z1[h > 0.5], seed=seed)
                                     - h[h > 0.5]) / eps
                    dhdz[h > 0.5] = (double_perlin3D(
                        x1[h > 0.5], y1[h > 0.5], z1[h > 0.5] + eps, seed=seed)
                                     - h[h > 0.5]) / eps

                    norm = normal(ox1, y1, oz1, h, dhdx, dhdy, dhdz, A=0.05)

                    i_var = np.ones_like(h)

                    i_var[h > 0.5] = (0.5 + double_perlin3D(
                        x1[h > 0.5],
                        y1[h > 0.5],
                        z1[h > 0.5],
                        a=20,
                        d=5,
                        persistance=0.8,
                        seed=seed + 214))

                    im_rgbref_base[i, ok, d1, d2, 0] = rgbref[:, 0] * i_var
                    im_rgbref_base[i, ok, d1, d2, 1] = rgbref[:, 1] * i_var
                    im_rgbref_base[i, ok, d1, d2, 2] = rgbref[:, 2] * i_var
                    im_rgbref_base[i, ok, d1, d2, 3] = rgbref[:, 3]

                    normal_base[i, ok, d1, d2, 0] = norm[0]
                    normal_base[i, ok, d1, d2, 1] = norm[1]
                    normal_base[i, ok, d1, d2, 2] = norm[2]

                im_rgbref_base[i, ~ok, d1, d2, 0] = background(
                    x[i, ~ok], y[i, ~ok])[0, :]
                im_rgbref_base[i, ~ok, d1, d2, 1] = background(
                    x[i, ~ok], y[i, ~ok])[1, :]
                im_rgbref_base[i, ~ok, d1, d2, 2] = background(
                    x[i, ~ok], y[i, ~ok])[2, :]

    return im_rgbref_base, normal_base, cloud_map


def lum_perc(cos):
    u = np.log(1 + 5 * np.abs(cos)) / np.log(1 + 5)
    return u


def apply_illum(im_rgbref_base, normal_base, cloud_map, ilum=(1, 0.5, 1)):
    im = np.zeros((res, res, 3))
    for i in range(res):
        for d1 in range(multi_sampling):
            for d2 in range(multi_sampling):
                z2 = radius**2 - (x[i, :] + dx[d1, d2])**2 - (
                    y[i, :] + dy[d1, d2])**2
                ok = z2 >= 0

                if np.any(ok):
                    x1, y1, z1 = (x[i, ok] + dx[d1, d2]), (
                        y[i, ok] + dy[d1, d2]), np.sqrt(z2[ok])

                    cos1 = cos_vec(ilum, (x1, y1, z1))
                    norm = normal_base[i, ok, d1, d2, 0], normal_base[
                        i, ok, d1, d2, 1], normal_base[i, ok, d1, d2, 2]
                    cos = np.maximum(cos_vec(ilum, norm), cos1 / 2)

                    refvec = reflection((0, 0, -1), norm)
                    refillum = np.where(
                        cos1 >= 0,
                        np.maximum(cos_vec(refvec, ilum), 0)**100, 0)

                    u = np.where(cos > 0, lum_perc(cos), 0) / multi_sampling**2
                    q = refillum * im_rgbref_base[i, ok, d1, d2,
                                                  3] / multi_sampling**2

                    r_sol = im_rgbref_base[i, ok, d1, d2, 0] * u + q
                    g_sol = im_rgbref_base[i, ok, d1, d2, 1] * u + q
                    b_sol = im_rgbref_base[i, ok, d1, d2, 2] * u + q

                    u = np.where(cos1 > 0, lum_perc(cos1),
                                 0) / multi_sampling**2
                    f = np.exp(
                        -14 * (np.clip(cloud_map[i, ok, d1, d2], 0.5, 1) - .5))

                    im[i, ok, 0] += r_sol * f + (1 - f) * u
                    im[i, ok, 1] += g_sol * f + (1 - f) * u
                    im[i, ok, 2] += b_sol * f + (1 - f) * u

                xx1 = x[i, ~ok] + dx[d1, d2]
                yy1 = y[i, ~ok] + dy[d1, d2]
                atmos_dens = np.exp(-(np.sqrt(xx1**2 + yy1**2) - radius) /
                                    (atmos_size * radius))

                atmos_illum = atmos_dens * (np.maximum(
                    cos_vec(ilum, (xx1, yy1, 0)), 0))

                im[i, ~ok, 0] += (
                    im_rgbref_base[i, ~ok, d1, d2, 0] +
                    atmos_illum * atmos_color[0]) / multi_sampling**2
                im[i, ~ok, 1] += (
                    im_rgbref_base[i, ~ok, d1, d2, 1] +
                    atmos_illum * atmos_color[1]) / multi_sampling**2
                im[i, ~ok, 2] += (
                    im_rgbref_base[i, ~ok, d1, d2, 2] +
                    atmos_illum * atmos_color[2]) / multi_sampling**2

    np.clip(im, 0, 1, out=im)
    return im


if __name__ == '__main__':
    for seed in range(0, 100):
        a, b, c = gen_rgbbase_and_normal(seed=seed)
        im = apply_illum(a, b, c)
        plt.axis("off")
        plt.imshow(im)
        plt.imsave("exemples/out"+str(seed)+".png", im)
        plt.show()
