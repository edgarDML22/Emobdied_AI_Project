#
#   python_imagenes_cargar.py
#
#   Carga una imagen usando OpenCV y la muestra en una ventana
#
#   Para usar OpenCV
#       pip3 install opencv-python
#   Se requiere la imagen "robot.jpg" (o cualquier otra)
#
#   Rogelio Ferreira Escutia - marzo 2024
#

# Cargar OpenCV
import cv2 as cv

# Imprimir versión de OpenCV
print("Versión de OpenCV:", cv.__version__)

# Cargar imagen a memoria 
#   0=escala de grises
#   1=rgb
img = cv.imread('src/session_1/media/robot.jpg', 1)

# Crear una ventana de salida
cv.imshow('ejemplo', img)

# Esperar por una tecla
cv.waitKey(0)

# Cerrar la ventana
cv.destroyAllWindows()