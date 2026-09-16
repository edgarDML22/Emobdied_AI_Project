# Task t02 - Detección de personas y objetos en video
# Edgar Daniel Martínez López - 16/09/2026

import cv2
from ultralytics import YOLO

def detectar_en_video(video_path: str):
    # Cargar el modelo remoto
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en '{video_path}'")
        return

    print("Reproduciendo video con detecciones. Presiona 'q' o 'ESC' para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o no se pudo leer el fotograma.")
            break

        # Inferencia con YOLO (stream=True optimiza memoria para streaming de video)
        results = model(frame, stream=True, conf=0.45)

        for r in results:
            for box in r.boxes:
                # Coordenadas
                x1, y1, x2, y2 = box.xyxy[0].astype(int)

                # Confianza y clase
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{model.names[cls_id]} {conf:.2f}"

                # Si solo te interesara detectar personas, bastaría con:
                # if model.names[cls_id] != 'person': continue

                # Bounding box y texto
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 105, 180), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

        cv2.imshow("Deteccion de Objetos - Video", frame)

        # Salir con ESC (27) o la tecla 'q'
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Coloca aquí el nombre exacto de tu archivo en src/session_1/videos/
    ruta_video = "src/session_1/videos/people.mp4"
    detectar_en_video(ruta_video)