import numpy as np

from geometry import dot


def fract(x):
    return x - np.floor(x)


def hash(x, y=0, z=0):
    return hash13(fract(x * .1031), fract(y * .1031), fract(z * .1031))


def hash13(p3x, p3y, p3z):
    s = p3x * (p3y + 19.19) + p3y * (p3z + 19.19) + p3z * (p3x + 19.19)
    p3x = p3x + s
    p3y = p3y + s
    p3z = p3z + s
    return fract((p3x + p3y) * p3z)


def hash83(p3x, p3y, p3z, p3X, p3Y, p3Z):
    p3xx = (p3x + 19.19)
    p3yy = (p3y + 19.19)
    p3zz = (p3z + 19.19)
    p3XX = (p3X + 19.19)
    p3YY = (p3Y + 19.19)
    p3ZZ = (p3Z + 19.19)

    uxy = p3x * p3yy
    uyz = p3y * p3zz
    uzx = p3z * p3xx

    uxY = p3x * p3YY
    uyZ = p3y * p3ZZ
    uzX = p3z * p3XX

    uXy = p3X * p3yy
    uYz = p3Y * p3zz
    uZx = p3Z * p3xx

    uXY = p3X * p3YY
    uYZ = p3Y * p3ZZ
    uZX = p3Z * p3XX

    sxyz = uxy + uyz + uzx
    sXyz = uXy + uyz + uzX
    sxYz = uxY + uYz + uzx
    sXYz = uXY + uYz + uzX
    sxyZ = uxy + uyZ + uZx
    sXyZ = uXy + uyZ + uZX
    sxYZ = uxY + uYZ + uZx
    sXYZ = uXY + uYZ + uZX

    nxy = p3x + p3y
    nXy = p3X + p3y
    nxY = p3x + p3Y
    nXY = p3X + p3Y

    vxyz = fract((nxy + sxyz + sxyz) * (p3z + sxyz))
    vXyz = fract((nXy + sXyz + sXyz) * (p3z + sXyz))
    vxYz = fract((nxY + sxYz + sxYz) * (p3z + sxYz))
    vXYz = fract((nXY + sXYz + sXYz) * (p3z + sXYz))
    vxyZ = fract((nxy + sxyZ + sxyZ) * (p3Z + sxyZ))
    vXyZ = fract((nXy + sXyZ + sXyZ) * (p3Z + sXyZ))
    vxYZ = fract((nxY + sxYZ + sxYZ) * (p3Z + sxYZ))
    vXYZ = fract((nXY + sXYZ + sXYZ) * (p3Z + sXYZ))

    return vxyz, vXyz, vxYz, vXYz, vxyZ, vXyZ, vxYZ, vXYZ


def polynom_interpol(x):
    x2 = x * x
    x4 = x2 * x2
    return 6 * x4 * x - 15 * x4 + 10 * x2 * x


def noise3D(x, y, z, seed=42):
    u = hash(seed)
    x = x + u * 1431.26742
    y = y + u * 3262.42321
    z = z + u * 7929.27883

    ix, iy, iz = np.floor(x), np.floor(y), np.floor(z)

    fx, fy, fz = x - ix, y - iy, z - iz

    wx = polynom_interpol(fx)
    wy = polynom_interpol(fy)
    wz = polynom_interpol(fz)

    p3x = fract(ix * .1031)
    p3y = fract(iy * .1031)
    p3z = fract(iz * .1031)
    p3X = fract((ix + 1) * .1031)
    p3Y = fract((iy + 1) * .1031)
    p3Z = fract((iz + 1) * .1031)

    vxyz, vXyz, vxYz, vXYz, vxyZ, vXyZ, vxYZ, vXYZ = hash83(
        p3x, p3y, p3z, p3X, p3Y, p3Z)

    a1 = vxyz + (vXyz - vxyz) * wx
    a2 = vxYz + (vXYz - vxYz) * wx
    a3 = vxyZ + (vXyZ - vxyZ) * wx
    a4 = vxYZ + (vXYZ - vxYZ) * wx

    b1 = a1 + (a2 - a1) * wy
    b2 = a3 + (a4 - a3) * wy

    c = b1 + (b2 - b1) * wz

    return c


def perlin3D(x, y, z, octave=7, persistance=0.5, seed=42):
    p = 1
    totp = 0
    tot = 0

    for _ in range(octave):
        c = noise3D(x, y, z, seed)

        tot += p * c
        totp += p

        p *= persistance

        x, y, z = x * 2, y * 2, z * 2

    return tot / totp


def double_perlin3D(x, y, z, a=4, d=2, octave=8, persistance=0.6, seed=42):
    dx = perlin3D(x, y, z, 5, 0.5, seed * hash(seed, 0, 10))
    dy = perlin3D(x, y, z, 5, 0.5, seed * hash(seed, 10, 0))
    dz = perlin3D(x, y, z, 5, 0.5, seed * hash(seed, 10, 10))

    dx, dy, dz = d * (dx - 0.5), d * (dy - 0.5), d * (dz - 0.5)

    P = perlin3D(a * x + dx, a * y + dy, a * z + dz, octave, persistance, seed)

    return P


def field(x, y, z, t):
    strength = 7. + .03 * np.log(1.e-6 + fract(np.sin(t) * 4373.11))
    accum = np.zeros_like(x)
    prev = np.zeros_like(x)
    tw = np.zeros_like(x)
    p = x, y, z
    for i in range(32):
        mag = dot(p, p)
        p = np.abs(p[0]) / mag, np.abs(p[1]) / mag, np.abs(p[2]) / mag
        p = p[0] - .51, p[1] - .4, p[2] - 1.3
        w = np.exp(-i / 7.)
        accum += w * np.exp(-strength * (np.abs(mag - prev)**2.3))
        tw += w
        prev = mag

    return np.maximum(0., 5. * accum / tw - .7)


def background(x, y, t=437.2):
    zeros = np.zeros_like(x)
    p = np.array([x / 4 + 2, y / 4 - 1.3, -1 + zeros])
    p += 0.15 * (np.array([zeros + 1.0, zeros + 0.5, zeros + np.sin(t / 128.)
                           ]))

    t1 = field(p[0, :], p[1, :], p[2, :], t)
    v = (1. - np.exp((np.abs(x) - 1.) * 6.)) * (1. - np.exp(
        (np.abs(y) - 1.) * 6.))

    c1 = (.4 + .6 * v) * np.array([1.8 * t1 * t1 * t1, 1.4 * t1 * t1, t1])
    return c1
