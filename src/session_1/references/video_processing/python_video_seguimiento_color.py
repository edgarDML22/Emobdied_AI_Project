#
#   python_video_seguimiento_color.py
#
#   Carga un video usando OpenCV y hace un seguimiento por color
#
#       Para usar OpenCV: pip3 install opencv-python
#       Se requiere una cámara y darle permiso de acceso
#
#   Rogelio Ferreira Escutia - marzo 2024
#

import cv2 as cv 

cap = cv.VideoCapture(0)
while(True):
    ret, img = cap.read()
    if ret == True:
        #cv.imshow('video', img)
        imghsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # Deteccion de colores cálidos
        ubb=(0,100, 100)
        uba=(20, 255,255)
        # Deteccion de colores azules (azules)
        # ubb = (90, 100, 100)
        # uba = (130, 255, 255)
        # Deteccion de colores verdes 
        # ubb = (35, 100, 100)	
        # uba = (85, 255, 255)
        mask = cv.inRange(imghsv, ubb, uba)
        res = cv.bitwise_and(img, img, mask=mask)
        cv.imshow('resultado', res)

        k =cv.waitKey(1) & 0xFF
        if k == 27 :
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
