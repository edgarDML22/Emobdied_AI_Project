# Tarea T01: Detección de rostros
# Edgar Daniel Martínez López - Octubre 2026

import os
import urllib.request
import cv2


def descargar_modelo_si_no_existe(ruta_modelo, url):
    """Descarga el modelo desde la URL dada si no se encuentra localmente."""
    if not os.path.exists(ruta_modelo):
        print(f"Modelo '{ruta_modelo}' no encontrado.")
        print("Iniciando descarga automática desde OpenCV Zoo (puede tardar unos segundos)...")
        try:
            urllib.request.urlretrieve(url, ruta_modelo)
            print(f"¡Descarga de '{ruta_modelo}' completada con éxito!")
        except Exception as e:
            print(f"Error al descargar el modelo: {e}")
            return False
    return True


def inicializar_detector_yunet(model_path="face_detection_yunet_2023mar.onnx", conf_threshold=0.8, nms_threshold=0.3):
    """Inicializa el detector ultra ligero basado en YuNet (OpenCV DNN)"""
    url_yunet = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    if not descargar_modelo_si_no_existe(model_path, url_yunet):
        return None

    return cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(320, 320),
        score_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU
    )


def detectar_rostros(img, detector):
    """Ejecuta la inferencia sobre el frame y retorna la imagen dibujada y las detecciones crudas"""
    if detector is None:
        return img, None

    h, w, _ = img.shape
    detector.setInputSize((w, h))

    _, faces = detector.detect(img)

    if faces is not None:
        for face in faces:
            x, y, w_box, h_box = face[0:4].astype(int)
            conf = float(face[14])

            cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(img, f"Face: {conf:.2f}", (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img, faces
