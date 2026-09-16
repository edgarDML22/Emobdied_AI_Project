# Task T03
# - T03: Detección de rostros en una imagen (No entregar nada).
# Edgar Daniel Martínez López - 16/09/2026

import os
import cv2
import numpy as np


class FaceDetection:
    def __init__(self, model_path="face_detection_yunet_2023mar.onnx", conf_threshold=0.8, nms_threshold=0.3, use_cuda=False):
        """
        Detector de rostros ultra ligero basado en YuNet (OpenCV DNN).
        Ideal para despliegue en Edge / Jetson Orin Nano.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontro el modelo en '{model_path}'. "
                f"Descárgalo desde el repositorio oficial de OpenCV Zoo."
            )

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # Configurar backend: CPU por defecto para dejar la GPU libre a navegación/SLAM
        backend_id = cv2.dnn.DNN_BACKEND_CUDA if use_cuda else cv2.dnn.DNN_BACKEND_OPENCV
        target_id = cv2.dnn.DNN_TARGET_CUDA if use_cuda else cv2.dnn.DNN_TARGET_CPU

        # Inicializar FaceDetectorYN
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),  
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000,
            backend_id=backend_id,
            target_id=target_id
        )

    def detect(self, img):
        """
        Ejecuta la inferencia sobre una imagen o frame.
        Retorna: Lista de diccionarios con 'box', 'confidence' y 'landmarks'.
        """
        h, w, _ = img.shape
        # YuNet requiere conocer la resolución exacta del frame antes de inferir
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(img)
        detections = []

        if faces is not None:
            for face in faces:
                # Estructura del vector YuNet:
                # [0:4]   -> x, y, w, h
                # [4:14]  -> 5 landmarks: (ojo_der, ojo_izq, nariz, boca_der, boca_izq)
                # [14]    -> confianza de detección
                x, y, w_box, h_box = face[0:4].astype(int)
                # Column Vectors [[ojo_der(x), ojo der(y)], 
                # [ojo_izq(x), ojo_izq(y)]...]
                landmarks = face[4:14].reshape((5, 2)).astype(int)
                
                # Confidence output from the model
                confidence = float(face[14])

                detections.append({
                    "box": (x, y, w_box, h_box),
                    "confidence": confidence,
                    "landmarks": {
                        "right_eye": landmarks[0],
                        "left_eye": landmarks[1],
                        "nose": landmarks[2],
                        "right_mouth": landmarks[3],
                        "left_mouth": landmarks[4]
                    },
                    "raw_landmarks": landmarks
                })

        return detections

    def draw_faces(self, img, detections):
        """This method draws the bounding box and marks the 5
        landmarks that the model gave as a result"""
        # Colores utilizados para cada landkmark
        landmark_colors = [
            (255, 0, 0),    # Ojo derecho (Azul)
            (0, 0, 255),    # Ojo izquierdo (Rojo)
            (0, 255, 0),    # Nariz (Verde)
            (255, 0, 255),  # Comisura derecha (Magenta)
            (0, 255, 255)   # Comisura izquierda (Amarillo)
        ]

        for det in detections:
            x, y, w, h = det["box"]
            conf = det["confidence"]
            landmarks = det["raw_landmarks"]

            # 1. Bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 2. Etiqueta con confianza
            label = f"Face: {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_y = max(y, th + 10)
            
            cv2.rectangle(img, (x, text_y - th - 5), (x + tw, text_y + baseline - 5), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, label, (x, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # 3. Dibujar los 5 landmarks
            for point, color in zip(landmarks, landmark_colors):
                cv2.circle(img, tuple(point), 3, color, 3)

        return img

    def detect_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: No se encontro la imagen en '{image_path}'")
            return

        detections = self.detect(img)
        print(f"Rostros encontrados: {len(detections)}")

        output_img = self.draw_faces(img, detections)
        cv2.imshow("Deteccion de Rostros - YuNet", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_video(self, source='webcam', webcam_index=0, video_path=None, speed_multiplier=1.0):
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
            print("Error: No se pudo abrir la fuente de video.")
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

                detections = self.detect(frame)
                display_frame = self.draw_faces(frame, detections)
                cv2.imshow("Deteccion de Rostros en Tiempo Real", display_frame)

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


if __name__ == "__main__":
    detector = FaceDetection()
    
    # Probar con imagen fija:
    # detector.detect_image("foto.jpg")
    
    # Probar con webcam:
    detector.detect_video(source='webcam', webcam_index=0)