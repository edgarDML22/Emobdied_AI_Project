#
#   python_imagenes_filtro_convolucion.py
#
#   Carga una imagen usando OpenCV y le aplica un filtro convolucional para detectar bordes
#
#   Para usar OpenCV
#       pip3 install opencv-python
#   Se requiere la imagen "robot.jpg" (o cualquier otra)
#
#   Rogelio Ferreira Escutia - marzo 2024
#

import cv2 as cv 
import numpy as np 

img = cv.imread('src/session_1/media/robot.jpg',0)
mtz = np.array([[1,0,-1],
                [2,0,-2],
                [1,0,-1]])
resultado = cv.filter2D(img, -1, mtz)
cv.imshow('marcoc', resultado)
cv.imshow('marco', img)
cv.waitKey(0)
cv.destroyAllWindows()