# Tarea T01: Detección de rostros
# Edgar Daniel Martínez López - Octubre 2026

import os
import cv2

def inicializar_detector_yunet(model_path="face_detection_yunet_2023mar.onnx", conf_threshold=0.8, nms_threshold=0.3):
    """Inicializa el detector ultra ligero basado en YuNet (OpenCV DNN)"""
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en '{model_path}'.")
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
    if detector is None: return img, None
    
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