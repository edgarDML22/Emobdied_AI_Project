#
#   python_imagenes_canales_de_color.py
#
#   Carga una imagen usando OpenCV y muestra cada canal de color
#
#   Para usar OpenCV
#       pip3 install opencv-python
#   Se requiere la imagen "robot.jpg" (o cualquier otra)
#
#   Rogelio Ferreira Escutia - marzo 2024
#

# Cargar OpenCV
import cv2 as cv

# Cargar la biblioteca NumPy
import numpy as np

# Imprimir versión de OpenCV
print("Versión de OpenCV:", cv.__version__)

# Cargar imagen a memoria 
img = cv.imread('src/session_1/media/robot.jpg')

# Crea una matriz llena de ceros
img2 = np.zeros(img.shape[:2], dtype=np.uint8)

print(img.shape[1])
b, g, r = cv.split(img)

# img 2 es una matriz de ceros, al usar el método merge, coloca como activo el canal del color
# extraido de img original, y el resto de canales los coloca en cero.
# Imagen Cyan: Blue + Green
rb = cv.merge([b,g,img2])
# Imagen Yellow: Red + Green
rg = cv.merge([img2,g, r])
# Imagen Magenta: Red + Blue
rr = cv.merge([b,img2, r])

# Imagen original
cv.imshow('img', img)

# Canales de colores
cv.imshow('b', rb)
cv.imshow('g', rg)
cv.imshow('r', rr)

# Esperar por una tecla
cv.waitKey(0)

# Cerrar la ventana
cv.destroyAllWindows()