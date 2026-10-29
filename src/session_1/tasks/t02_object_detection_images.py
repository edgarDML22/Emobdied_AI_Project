# Task t02 - Detección de personas y objetos en imagen
# Edgar Daniel Martínez López - 16/09/2026

import cv2
import math
from ultralytics import YOLO

def detectar_en_imagen(image_path: str):
    # 1. Cargar el modelo remoto 
    # yolov8n.pt es la versión 'nano': la más ligera y rápida
    model = YOLO("yolov8n.pt")

    # 2. Cargar la imagen
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: There was not image found in the path: '{image_path}'")
        return

    # 3. Obtener el resultado del modelo
    # conf=0.4 descarta detecciones con menos del 40% de certeza
    results = model(img, conf=0.4)

    # 4. Procesar las detecciones
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            # Extraer coordenadas de la caja (x1, y1) arriba-izq, (x2, y2) abajo-der
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confianza redondeada a 2 decimales
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Clase detectada: model.names contiene el mapeo id -> nombre ("person", "car", etc.)
            cls_id = int(box.cls[0])
            # un dictionary con los nombres de las clases de la forma person: 0
            class_name = model.names[cls_id] 

            # Formatear label
            label = f"{class_name} {int(confidence * 100)}%"

            # Dibujar Bounding Box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Fondo para el texto
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)

            # Escribir la etiqueta
            cv2.putText(
                img, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 0), 
                1, 
                cv2.LINE_AA
            )

    # 5. Mostrar resultado
    cv2.imshow("Deteccion de Objetos - Imagen", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Ruta de prueba hacia tu imagen
    ruta_imagen = "src/session_1/media/farm.jpg"
    detectar_en_imagen(ruta_imagen)