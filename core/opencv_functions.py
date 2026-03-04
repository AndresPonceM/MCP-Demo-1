import cv2
import numpy as np
import os

def desenfocar_imagen(imagen_original, params):
    k = int(params.get("tamaño kernel"))
    gaussian = cv2.GaussianBlur(imagen_original, (k, k), 0)
    return gaussian
    
def brillo_contraste_imagen(imagen_original, params):
    alpha = float(params.get("contraste", 1.0))
    beta = int(params.get("brillo", 0))
    res = cv2.convertScaleAbs(imagen_original, alpha=alpha, beta=beta)
    return res
    