import numpy as np
import matplotlib.pyplot as plt

patrat = np.array([
    [0, 0, 1],
    [2, 0, 1],
    [2, 2, 1],
    [0, 2, 1]
], dtype=float)

poligon = np.array([
    [0, 0, 1],
    [3, 0, 1],
    [4, 1, 1],
    [2.5, 3, 1],
    [0.5, 2.5, 1],
    [0, 1, 1]
], dtype=float)


def aplica(M, puncte):
    return (M @ puncte.T).T


def translatie(mutare_x, mutare_y):
    return np.array([
        [1, 0, mutare_x],
        [0, 1, mutare_y],
        [0, 0, 1]
    ], dtype=float)


def scalare(factor_x, factor_y):
    return np.array([
        [factor_x, 0, 0],
        [0, factor_y, 0],
        [0, 0, 1]
    ], dtype=float)


def rotatie(unghi_grade):
    a = np.deg2rad(unghi_grade)
    c = np.cos(a)
    s = np.sin(a)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=float)


def reflexie_ox():
    return np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=float)


def reflexie_oy():
    return np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=float)


def shear(shear_x, shear_y):
    return np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0],
        [0, 0, 1]
    ], dtype=float)


def ploteaza(figura_init, figura_transf, titlu):
    plt.figure()
    x0 = figura_init[:, 0]
    y0 = figura_init[:, 1]
    x0 = np.append(x0, x0[0])
    y0 = np.append(y0, y0[0])
    plt.plot(x0, y0, "b-o", label="initial")

    x1 = figura_transf[:, 0]
    y1 = figura_transf[:, 1]
    x1 = np.append(x1, x1[0])
    y1 = np.append(y1, y1[0])
    plt.plot(x1, y1, "r-o", label="transformat")

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.gca().set_aspect("equal", "box")
    plt.grid(True)
    plt.title(titlu)
    plt.legend()
    plt.show()


mutare_x = float(input("Translatie - mutare_x = "))
mutare_y = float(input("Translatie - mutare_y = "))
M = translatie(mutare_x, mutare_y)
ploteaza(patrat, aplica(M, patrat), f"Translatie ({mutare_x}, {mutare_y})")

factor_x = float(input("Scalare - factor_x = "))
factor_y = float(input("Scalare - factor_y = "))
M = scalare(factor_x, factor_y)
ploteaza(patrat, aplica(M, patrat), f"Scalare ({factor_x}, {factor_y})")

unghi = float(input("Rotatie - unghi = "))
M = rotatie(unghi)
ploteaza(patrat, aplica(M, patrat), f"Rotatie ({unghi})")

M = reflexie_ox()
ploteaza(patrat, aplica(M, patrat), "Reflexie fata de Ox")

M = reflexie_oy()
ploteaza(patrat, aplica(M, patrat), "Reflexie fata de Oy")

shear_x = float(input("Shear - shear_x = "))
shear_y = float(input("Shear - shear_y = "))
M = shear(shear_x, shear_y)
ploteaza(poligon, aplica(M, poligon), f"Shear (shx={shear_x}, shy={shear_y})")
