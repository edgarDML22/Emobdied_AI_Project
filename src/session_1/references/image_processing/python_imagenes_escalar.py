#
#   python_imagenes_escalar.py
#
#   Carga una imagen usando OpenCV y la escala en un factor de 2
#
#   Para usar OpenCV
#       pip3 install opencv-python
#   Se requiere la imagen "robot.jpg" (o cualquier otra)
#
#   Rogelio Ferreira Escutia - marzo 2024
#

import cv2 as cv    # Cargar OpenCV
import numpy as np  # Cargar NumPy

# Imprimir versión de OpenCV
print("Versión de OpenCV:", cv.__version__)

# Cargar imagen a memoria 
#   0=escala de grises
#   1=rgb
img = cv.imread('src/session_1/media/robot.jpg',0)

# Crea una matriz llena de unos
img2 = np.ones((img.shape[0]*2, img.shape[1]*2), dtype=np.uint8)


for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img2[i*2][j*2]=img[i][j]
        
img3 = cv.resize(img, (0, 0), fx=2, fy=2, interpolation=cv.INTER_CUBIC)
# Los 3 metodos de interpolacion más utilizados son: INTER_NEAREST (más utilizado) , INTER_LNEAR, INTER_CUBIC (más costoso)

cv.imshow('Original', img)
cv.imshow('Prueba 1', img2)
cv.imshow('Prueba 2', img3)


# Esperar por una tecla
cv.waitKey(0)

# Cerrar la ventana
cv.destroyAllWindows()