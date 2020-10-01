import numpy as np


def dot(a, b):
    x1, y1, z1 = a
    x2, y2, z2 = b
    return x1 * x2 + y1 * y2 + z1 * z2


def cos_vec(a, b):
    return dot(a, b) / np.sqrt(dot(a, a) * dot(b, b))


def cross(a, b):
    x1, y1, z1 = a
    x2, y2, z2 = b
    return (y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2)


def reflection(d, n):
    xd, yd, zd = d
    xn, yn, zn = n

    u = dot(d, n) / dot(n, n)

    return (xd - 2 * u * xn, yd - 2 * u * yn, zd - 2 * u * zn)


def normal(x, y, z, h, dx, dy, dz, A=0.01):
    '''
    x = R cos phi cos theta
    y = R sin phi cos theta
    z = R sin theta
    '''

    R = np.sqrt(x * x + y * y + z * z)
    costheta = np.sqrt(x * x + y * y) / R
    sintheta = z / R
    cosphi = x / (R * costheta)
    sinphi = y / (R * costheta)
    '''
    r = R (1 + A*h)
    '''
    r = R * (1 + A * h)

    drdphi = R * A * (dx * -y + dy * x)
    drdtheta = R * A * (dx * (-z * cosphi) + dy *
                        (-z * cosphi) + dz * R * costheta)

    uphi = (r * -sinphi * costheta + drdphi * cosphi * costheta,
            r * cosphi * costheta + drdphi * sinphi * costheta, 0)

    utheta = (r * cosphi * -sintheta + drdtheta * cosphi * costheta,
              r * sinphi * -sintheta + drdtheta * sinphi * costheta,
              r * costheta + drdtheta * sintheta)

    v = cross(uphi, utheta)
    nv = dot(v, v)
    v = v[0] / nv, v[1] / nv, v[2] / nv
    return v
