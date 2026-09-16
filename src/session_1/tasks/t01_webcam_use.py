# Task t01
# - T01: Extraer de una Webcam, el streaming de video y mostrarlo en una ventana 
# Edgar Daniel Martínez López - 16/09/2026
# 

import cv2

# Este index permite identificar a que dispositivo nos estamos conectando
# Si tenemos más de un dispositivo podemos ejecutar el script t01_device_detector
# para identificar el index asignado a cada dispositivo de entrada.
cap = cv2.VideoCapture(0)

while True:
    # Este método combina 2 operaciones cap.grap() y cap.retrieve()
    # cap.grab() obtiene la informacion obtiene el siguiente fotograma al input device 
    # cap.retrieve() decodifica el fotograma a una matriz de Numpy
    # devolviendo 2 valores
    # 1. ret: return value (boolean -> para saber si s obtuvo el fotograma
    # 2. img (np.array): la imagen obtenida o None
    ret, img = cap.read()
    if ret == True:
        cv2.imshow('Video de la Webcam', img)
        
        # Máscara AND de '11111111' para obtener valor dentro de rango ASCII
        k = cv2.waitKey(1) & 0xFF
        
        # 27 corresponde a la tecla ESC en ASCIII
        if k == 27: break
    else: break      

# Liberar input device
cap.release()

cv2.destroyAllWindows()