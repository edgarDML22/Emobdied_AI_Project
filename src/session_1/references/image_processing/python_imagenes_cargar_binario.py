#
#   python_imagenes_cargar_binario.py
#
#   Carga una imagen usando OpenCV y la muestra en binario
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

# Crea una matriz llena de ceros
img2 = np.zeros(img.shape[:2], dtype=np.uint8)


for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (img[i][j]>150):
            img2[i][j]=255
        else:
            img2[i][j]=0
# Este for lo que hace es que si el pixel de la posición [i][j] tiene un valor > 150 lo pinta de blanco
# Sino, lo pinta de negro
cv.imshow('Prueba', img)
cv.imshow('Prueba', img2)

# Esperar por una tecla
cv.waitKey(0)

# Cerrar la ventana
cv.destroyAllWindows()