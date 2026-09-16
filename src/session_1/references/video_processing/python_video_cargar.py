#
#   python_video_cargar.py
#
#   Carga un video usando OpenCV y lo muestra en una ventana
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
        cv.imshow('video', img)
        k = cv.waitKey(1) & 0xFF
        if k ==27:
            break
    else:
        break
cap.release()

# Cerrar la ventana
cv.destroyAllWindows()