#
#   python_video_cargar_canales.py
#
#   Carga un video usando OpenCV y lo muestra en una ventana el original y con sus canales RGB
#
#       Para usar OpenCV: pip3 install opencv-python
#       Se requiere una cámara y darle permiso de acceso
#
#   Rogelio Ferreira Escutia - marzo 2024
#

import cv2 as cv    # Cargar OpenCV
import numpy as np  # Cargar NumPy

# Imprimir versión de OpenCV
print("Versión de OpenCV:", cv.__version__)

cap = cv.VideoCapture(0)
while(True):
    ret, img = cap.read()
    if ret == True:
        
        # Crea una matriz llena de ceros
        img2 = np.zeros(img.shape[:2], dtype=np.uint8)

        b, g, r = cv.split(img)
        rb = cv.merge([b, img2, img2])
        rg = cv.merge([img2, g, img2])
        rr = cv.merge([img2, img2, r])

        cv.imshow('b', rb)
        cv.imshow('g', rg)
        cv.imshow('r', rr)

        cv.imshow('video', img)

        k = cv.waitKey(1) & 0xFF
        if k ==27:
            break
    else:
        break
cap.release()

# Cerrar la ventana
cv.destroyAllWindows()