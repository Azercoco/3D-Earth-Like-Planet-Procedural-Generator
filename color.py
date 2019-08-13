import numpy as np

biome_deep_water = {
    "maxA": 0,
    "minA": None,
    "A": -6000,
    "vA": 3000,
    "T": 40,
    "vT": 30,
    "H": None,
    "vH": None,
    "r": 16,
    "g": 25,
    "b": 36,
    "ref": 0.6
}
biome_median_water = {
    "maxA": 0,
    "minA": None,
    "A": -1000,
    "vA": 2000,
    "T": 40,
    "vT": 30,
    "H": None,
    "vH": None,
    "r": 16,
    "g": 25,
    "b": 36,
    "ref": 0.6
}
biome_shallow_water = {
    "maxA": 0,
    "minA": None,
    "A": -100,
    "vA": 300,
    "T": 40,
    "vT": 30,
    "H": None,
    "vH": None,
    "r": 50,
    "g": 79,
    "b": 114,
    "ref": 0.6
}


biome_ice = {
    "maxA": None,
    "minA": 0,
    "A": None,
    "vA": None,
    "T": -10,
    "vT": 5,
    "H": None,
    "vH": None,
    "r": 255,
    "g": 255,
    "b": 255,
    "ref": 0.6
}
biome_water_ice = {
    "maxA": 0,
    "minA": None,
    "A": -500,
    "vA": 500,
    "T": -15,
    "vT": 5,
    "H": None,
    "vH": None,
    "r": 200,
    "g": 200,
    "b": 255,
    "ref": 0.6
}

biome_cold_ice = {
    "maxA": None,
    "minA": 0,
    "A": None,
    "vA": None,
    "T": -40,
    "vT": 20,
    "H": None,
    "vH": None,
    "r": 255,
    "g": 255,
    "b": 255,
    "ref": 0.6
}
biome_cold_water_ice = {
    "maxA": 0,
    "minA": None,
    "A": None,
    "vA": None,
    "T": -45,
    "vT": 20,
    "H": None,
    "vH": None,
    "r": 200,
    "g": 200,
    "b": 255,
    "ref": 0.6
}

biome_tundra = {
    "maxA": None,
    "minA": 0,
    "A": 0,
    "vA": 2000,
    "T": -5,
    "vT": 5,
    "H": 0.5,
    "vH": 2,
    "r": 93,
    "g": 80,
    "b": 35,
    "ref": 0
}

biome_boreal_forest = {
    "maxA": None,
    "minA": 0,
    "A": 500,
    "vA": 1500,
    "T": 5,
    "vT": 5,
    "H": 2,
    "vH": 2,
    "r": 12,
    "g": 63,
    "b": 7,
    "ref": 0
}

biome_forest = {
    "maxA": None,
    "minA": 0,
    "A": 500,
    "vA": 1500,
    "T": 15,
    "vT": 5,
    "H": 2.5,
    "vH": 3,
    "r": 67,
    "g": 88,
    "b": 29,
    "ref": 0
}

biome_cold_desert = {
    "maxA": None,
    "minA": 0,
    "A": 500,
    "vA": 1500,
    "T": 10,
    "vT": 7,
    "H": 0,
    "vH": 1,
    "r": 245,
    "g": 200,
    "b": 151,
    "ref": 0
}
biome_warm_desert = {
    "maxA": None,
    "minA": 0,
    "A": 500,
    "vA": 1500,
    "T": 25,
    "vT": 7,
    "H": 0,
    "vH": 1,
    "r": 245,
    "g": 200,
    "b": 151,
    "ref": 0
}

biome_savanah = {
    "maxA": None,
    "minA": 0,
    "A": 0,
    "vA": 2000,
    "T": 25,
    "vT": 5,
    "H": 1,
    "vH": 1.5,
    "r": 66,
    "g": 105,
    "b": 13,
    "ref": 0
}
biome_humid_savanah = {
    "maxA": None,
    "minA": 0,
    "A": 0,
    "vA": 2000,
    "T": 25,
    "vT": 5,
    "H": 2,
    "vH": 1.5,
    "r": 53,
    "g": 112,
    "b": 46,
    "ref": 0
}
biome_jungle = {
    "maxA": None,
    "minA": 0,
    "A": 500,
    "vA": 1000,
    "T": 25,
    "vT": 5,
    "H": 3,
    "vH": 1.5,
    "r": 5,
    "g": 50,
    "b": 5,
    "ref": 0
}

biome_alpine_tundra = {
    "maxA": None,
    "minA": 0,
    "A": 3000,
    "vA": 1500,
    "T": 0,
    "vT": 5,
    "H": 0.5,
    "vH": 3,
    "r": 131,
    "g": 122,
    "b": 75,
    "ref": 0
}
biome_high_moutain = {
    "maxA": None,
    "minA": 0,
    "A": 6000,
    "vA": 2000,
    "T": None,
    "vT": None,
    "H": None,
    "vH": None,
    "r": 114,
    "g": 97,
    "b": 71,
    "ref": 0
}

biome_list = []

biome_list.append(biome_deep_water)
biome_list.append(biome_median_water)
biome_list.append(biome_shallow_water)
biome_list.append(biome_ice)
biome_list.append(biome_cold_ice)
biome_list.append(biome_water_ice)
biome_list.append(biome_cold_water_ice)
biome_list.append(biome_tundra)
biome_list.append(biome_boreal_forest)
biome_list.append(biome_forest)
biome_list.append(biome_cold_desert)
biome_list.append(biome_warm_desert)
biome_list.append(biome_savanah)
biome_list.append(biome_humid_savanah)
biome_list.append(biome_jungle)
biome_list.append(biome_alpine_tundra)
biome_list.append(biome_high_moutain)

T_eq = 30
T_pole = -40


def humidity(theta):
    theta_deg = 90 * theta / (np.pi / 2)

    c1 = (5 / (1 + (theta_deg / 10)**2))
    c2 = 2.5 / (1 + ((theta_deg - 50) / 10)**2)
    c3 = 2.5 / (1 + ((theta_deg + 50) / 10)**2)

    return 4 * (c1 + c2 + c3 - 0.5) / 4.692


def colorize(x, y, z, c, vT, vH):
    A = np.zeros_like(c)
    A[c > 0.5] = ((c[c > 0.5] - 0.5) / 0.5) * 8000  # land
    A[c < 0.5] = ((c[c < 0.5] - 0.5) / 0.5) * 12000  # ocean

    theta = np.arcsin(y / np.sqrt(x**2 + y**2 + z**2))

    T = T_pole + np.cos(theta) * (T_eq - T_pole) + (vT - 0.5) * 20
    T[c > 0.5] -= 7 * A[c > 0.5] / 1000

    H = humidity(theta) * (1 + (vH - 0.5) )

    rgbref = np.zeros((len(c), 4))
    pond_tot = np.zeros_like(c)

    for biome in biome_list:
        ok = np.ones_like(pond_tot, dtype=np.bool)

        if biome['minA'] is not None:
            ok = ok & (A >= biome['minA'])
        if biome['maxA'] is not None:
            ok = ok & (A <= biome['maxA'])

        pond = np.ones_like(pond_tot)

        if biome['A'] is not None:
            pond[ok] *= np.exp(-((A[ok] - biome['A']) / biome['vA'])**2)
        if biome['H'] is not None:
            pond[ok] *= np.exp(-((H[ok] - biome['H']) / biome['vH'])**2)
        if biome['T'] is not None:
            pond[ok] *= np.exp(-((T[ok] - biome['T']) / biome['vT'])**2)

        rgbref[ok, 0] += pond[ok] * biome['r']
        rgbref[ok, 1] += pond[ok] * biome['g']
        rgbref[ok, 2] += pond[ok] * biome['b']
        rgbref[ok, 3] += pond[ok] * biome['ref']

        pond_tot[ok] += pond[ok]

    rgbref[:, 0] /= (255 * pond_tot)
    rgbref[:, 1] /= (255 * pond_tot)
    rgbref[:, 2] /= (255 * pond_tot)
    rgbref[:, 3] /= (pond_tot)

    return rgbref
