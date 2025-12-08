import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from operatii import *

import pygame
WIDTH, HEIGHT = 800, 600
SCALE = 100


a = np.array([1.0, 1.0])
b = np.array([4.0, 1.0])
c = np.array([1.0, 3.0])

u = np.array([1.0, 0.0, 0.0])
v = np.array([0.0, 1.0, 0.0])
w = np.array([0.0, 0.0, 1.0])

def afisare_vectori(a,b,c,u,v,w):
    print("Vectori 2D:")
    print("a=", a)
    print("b=", b)
    print("c=", c)
    print("Vectori 3D:")
    print("u=", u)
    print("v=", v)
    print("w=", w)

def afisare_operatii(a,b,c,u,v,w):
    print("Norme: ")
    print("a= ", norma(a))
    print("b= ", norma(b))
    print("c= ", norma(c))
    print("u= ", norma(u))
    print("v= ", norma(v))
    print("w= ", norma(w))
    print("Sume si diferente 2D:")
    print("a+b= ", a+b)
    print("b+c= ", b+c)
    print("c-a= ", c-a)
    print("a-b= ", a-b)
    print("Sume si diferente 3D:")
    print("u+v= ", u+v)
    print("v+w= ", v+w)
    print("v-u= ", v-u)
    print("w-u= ", w-u)
    print("Produs scalar:")
    print("a * b= ", produs_scalar(a,b))
    print("b * c= ", produs_scalar(b,c))
    print("a * c= ", produs_scalar(a,c))
    print("u * v= ", produs_scalar(u,v))
    print("v * w= ", produs_scalar(v,w))
    print("u * w= ", produs_scalar(u,w))
    print("Unghiuri:")
    print("a,b= ", unghi(a,b))
    print("b,c= ", unghi(b,c))
    print("c,a= ", unghi(c,a))
    print("Produs vector:")
    print("u*v= ", produs_vector(u,v))
    print("v*w= ", produs_vector(v,w))
    print("u*w= ", produs_vector(u,w))
    print("Proiectie vectori:")
    print("a,b", proiectie(a,b))
    print("b,c", proiectie(b,c))
    print("c,a= ", proiectie(c,a))

def triunghi(a, b, c):
    pts = np.array([a, b, c, a])
    plt.figure()
    plt.plot(pts[:, 0], pts[:, 1], '-o')
    plt.title("Triunghi")
    plt.grid()
    plt.axis('equal')
    plt.show()

def dreptunghi(a, b, c):
    D = np.array([4.0, 3.0])
    A = a
    B = b
    C = c
    pts = np.array([A, B, D, C, A])
    plt.figure()
    plt.plot(pts[:, 0], pts[:, 1], '-o')
    plt.title("Dreptunghi")
    plt.grid()
    plt.axis('equal')
    plt.show()

def poligon(a, b, c):
    A = a
    B = b
    C = c
    AB = a + b
    AC = a + c
    pts = np.array([ A, B, AB, AC, C, A])
    plt.figure()
    plt.plot(pts[:, 0], pts[:, 1], '-o')
    plt.title("Poligon")
    plt.grid()
    plt.axis('equal')
    plt.show()

def cub(u, v, w):
    O = np.array([0.0, 0.0, 0.0])
    U = u
    V = v
    UV = u + v
    W = w
    UW = u + w
    VW = v + w
    UVW = u + v + w

    pts = np.array([O, U, V, UV, W, UW, VW, UVW])

    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, j in edges:
        xs = [pts[i, 0], pts[j, 0]]
        ys = [pts[i, 1], pts[j, 1]]
        zs = [pts[i, 2], pts[j, 2]]
        ax.plot(xs, ys, zs, '-o')

    ax.set_title("Paralelipiped")
    ax.set_box_aspect([1, 1, 1])
    plt.show()

def tetraedru(u, v, w):
    O = np.array([0.0, 0.0, 0.0])
    pts = np.array([O, u, v, w])

    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, j in edges:
        xs = [pts[i, 0], pts[j, 0]]
        ys = [pts[i, 1], pts[j, 1]]
        zs = [pts[i, 2], pts[j, 2]]
        ax.plot(xs, ys, zs, '-o')

    ax.set_title("Tetraedru")
    ax.set_box_aspect([1, 1, 1])
    plt.show()

def prisma(u, v, w):
    O = np.array([0.0, 0.0, 0.0])
    U = u
    V = v
    W = w
    UW = u + w
    VW = v + w

    pts = np.array([O, U, V, W, UW, VW])

    edges = [
        (0, 1), (1, 2), (2, 0),
        (3, 4), (4, 5), (5, 3),
        (0, 3), (1, 4), (2, 5)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, j in edges:
        xs = [pts[i, 0], pts[j, 0]]
        ys = [pts[i, 1], pts[j, 1]]
        zs = [pts[i, 2], pts[j, 2]]
        ax.plot(xs, ys, zs, '-o')

    ax.set_title("Prisma")
    ax.set_box_aspect([1, 1, 1])
    plt.show()

def vectori2d(a, b, c):
    plt.figure()
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1)
    plt.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1)
    plt.quiver(0, 0, c[0], c[1], angles='xy', scale_units='xy', scale=1)
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Vectori 2D")
    plt.grid()
    plt.show()

def vectori3d(u, v, w):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0, 0, 0, u[0], u[1], u[2])
    ax.quiver(0, 0, 0, v[0], v[1], v[2])
    ax.quiver(0, 0, 0, w[0], w[1], w[2])
    ax.set_title("Vectori 3D")
    ax.set_box_aspect([1, 1, 1])
    plt.show()


def proj_2d(p):
    x = int(WIDTH/2 + p[0] * SCALE)
    y = int(HEIGHT/2 - p[1] * SCALE)
    return x, y

def deseneaza_figuri_2d_pygame(screen):
    A = proj_2d(a)
    B = proj_2d(b)
    C = proj_2d(c)
    D = proj_2d(np.array([4.0, 3.0]))

    pygame.draw.polygon(screen, (200,50,50), [A,B,D,C], 2)
    pygame.draw.polygon(screen, (50,200,50), [A,B,C], 2)

def deseneaza_figuri_3d_pygame(screen):
    O = np.array([0,0])
    U = proj_2d(u[:2])
    V = proj_2d(v[:2])
    UV = proj_2d((u+v)[:2])

    pygame.draw.polygon(screen, (50,50,200), [U,V,UV], 2)


if __name__ == "__main__":
    afisare_vectori(a, b, c, u, v, w)
    afisare_operatii(a, b, c, u, v, w)

    triunghi(a, b, c)
    dreptunghi(a, b, c)
    poligon(a, b, c)
    cub(u, v, w)
    tetraedru(u, v, w)
    prisma(u, v, w)
    vectori2d(a, b, c)
    vectori3d(u, v, w)

