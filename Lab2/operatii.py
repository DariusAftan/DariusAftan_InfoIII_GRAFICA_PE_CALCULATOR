import numpy as np

def norma(v):
    return np.linalg.norm(v)

def produs_scalar(a, b):
    return np.dot(a, b)

def unghi(a,b):
    cos = produs_scalar(a,b) / (norma(a) * norma(b))
    return np.degrees(np.arccos(cos))

def produs_vector(u,v):
    return np.cross(u,v)

def proiectie(a,b):
    return (produs_scalar(a,b) / produs_scalar(b,b)) * b