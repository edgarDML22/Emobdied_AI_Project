# Task T02
# - T02: Detección de personas y objetos en una imagen (No entregar nada).
# Edgar Daniel Martínez López - 16/09/2026

import cv2
import random
import numpy as np
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self, model_path="yolov8n.pt"):
        # Inicializar modelo
        self.model = YOLO(model_path)
        
        # Generar un color único para cada clase
        random.seed(42)  # Semilla fija para mantener los mismos colores entre ejecuciones
        self.colors = {}
        for class_id in self.model.names.keys():
            self.colors[class_id] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )

    def __draw_bounding_boxes(self, img, results):
        """Draws the bounding boxes and label over the images"""
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Coordenadas [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Color asignado a la clase
                color = self.colors.get(class_id, (0, 255, 0))
                label = f"{class_name} {confidence:.2f}"
                
                # 1. Dibujar bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # 2. Calcular tamaño del texto y posición de la etiqueta
                (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Si la caja está pegada al borde superior, dibujar la etiqueta dentro de la caja
                text_y1 = max(y1, h + 10)
                
                # Rectangulo relleno para la label
                cv2.rectangle(
                    img, 
                    (x1, text_y1 - h - 5), 
                    (x1 + w, text_y1 + baseline - 5), 
                    color, 
                    cv2.FILLED
                )
                
                # Texto de la label
                cv2.putText(
                    img, 
                    label, 
                    (x1, text_y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 0), 
                    1, 
                    cv2.LINE_AA
                )
                
        return img  

    def object_detection_image(self, image_path, conf=0.45):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: No image found at '{image_path}'")
            return
        
        results = self.model(img, conf=conf)
        final_image = self.__draw_bounding_boxes(img, results)
        
        cv2.imshow("Deteccion de Objetos - Imagen", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        
    def object_detection_video(self, source='webcam', webcam_index=0, video_path=None, conf=0.45, speed_multiplier=1.0):
        """
        source: 'webcam', 'video'
        webcam_index = 0, 1, 2...
        speed_multiplier: Multiplicador inicial de velocidad de reproducción (0.5 a 2.0)
        """
        if source == 'webcam':
            cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
        elif source == 'video':
            if not video_path:
                print("Error: video_path must be provided when source='video'")
                return
            cap = cv2.VideoCapture(video_path)
        else:
            print(f"Error: Source '{source}' is not supported")
            return

        if not cap.isOpened():
            print(f"Error: Could not open source '{source}'")
            return

        # Configuración de velocidad y FPS para video
        speed_multiplier = max(0.5, min(2.0, float(speed_multiplier)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        base_delay = 1000.0 / fps

        is_paused = False
        display_frame = None

        while cap.isOpened():
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # stream=True optimiza el consumo de memoria en video continuo
                results = self.model(frame, conf=conf, stream=True)
                display_frame = self.__draw_bounding_boxes(frame, results)
                cv2.imshow("Detección de Objetos - Video", display_frame)

            # Control del tiempo de espera según la fuente y estado de pausa
            if source == 'video':
                delay = 30 if is_paused else max(1, int(base_delay / speed_multiplier))
            else:
                delay = 1

            k = cv2.waitKeyEx(delay)
            if k == -1:
                continue

            # Salir: ESC o 'q'/'Q'
            if k == 27 or (k & 0xFF == 27) or (k & 0xFF in (ord('q'), ord('Q'))):
                break
            # Pausa / Reanudar: Barra espaciadora
            elif k == 32 or (k & 0xFF == 32):
                if source == 'video':
                    is_paused = not is_paused
                    status = "Video pausado" if is_paused else "Video reanudado"
                    print(f"[{status}] - Velocidad: {speed_multiplier:.2f}x")
            # Flecha izquierda: Disminuir velocidad
            elif k in (2424832, 65361, 81, 37, 0x250000, 63234):
                if source == 'video':
                    speed_multiplier = max(0.5, round(speed_multiplier - 0.25, 2))
                    print(f"Velocidad: {speed_multiplier:.2f}x")
            # Flecha derecha: Aumentar velocidad
            elif k in (2555904, 65363, 83, 39, 0x270000, 63235):
                if source == 'video':
                    speed_multiplier = min(2.0, round(speed_multiplier + 0.25, 2))
                    print(f"Velocidad: {speed_multiplier:.2f}x")

        cap.release()
        cv2.destroyAllWindows()