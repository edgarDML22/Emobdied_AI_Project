#
#   python_webcam_deteccion_objetos_01.py
#
#   Detecta objetos por medio de una webcam utilizando YoLo
#
#       pip3 install opencv-python
#       pip3 install ultralytics
#
#   Rogelio Ferreira Escutia - diciembre 2023
#

# Bibliotecas a utilizar
from ultralytics import YOLO
import cv2
import math 

# Arranque de la webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Modelo de Yolo utilizado
model = YOLO("yolo-Weights/yolov8n.pt")

# Clases a identificar
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Coordenadas
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Coordenadas de la caja
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # Colocar la caja dentro de la cámara
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confianza
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # Impresión del nombre de la clase
            cls = int(box.cls[0]) 
            print("Class name -->", classNames[cls])

            # Detalles del objeto
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            print("Objeto: ", classNames[cls])

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()