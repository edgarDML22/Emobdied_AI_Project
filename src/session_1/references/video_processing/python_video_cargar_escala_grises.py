#
#   python_video_cargar_escala_grises.py
#
#   Carga un video usando OpenCV y lo muestra en una ventana el original y en escala de grises
#
#       Para usar OpenCV: pip3 install opencv-python
#       Se requiere una cámara y darle permiso de acceso
#
#   Rogelio Ferreira Escutia - marzo 2024
#

# Cargar OpenCV
import cv2 as cv

# Imprimir versión de OpenCV
print("Versión de OpenCV:", cv.__version__)

cap = cv.VideoCapture(0)
while(True):
    ret, img = cap.read()
    if ret == True:
        grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow('video', img)
        cv.imshow('Gray Scale Image Video', grayImage)
        k = cv.waitKey(1) & 0xFF
        if k ==27:
            break
    else:
        break
cap.release()

# Cerrar la ventana
cv.destroyAllWindows()