# Tarea T02: Registrar el rostro de una persona vía la Webcam y almacenar embeddings en ChromaDB
# Edgar Daniel Martínez López - Octubre 2026

import time
import uuid
import cv2
import chromadb
from t01_deteccion_rostros import inicializar_detector_yunet
from t03_reconocimiento_facial import inicializar_reconocedor_sface


def registrar_usuario(nombre_usuario, muestras_totales=150, webcam_index=0, retardo_segundos=0.1):
    """
    Registra embeddings faciales de 128 dimensiones en ChromaDB de forma persistente.
    Usa YuNet para detección y SFace para extracción de embeddings.
    """
    # 1. Inicialización de ChromaDB con manejo de errores
    try:
        cliente_chroma = chromadb.PersistentClient(path="./bd_vectores")
        coleccion = cliente_chroma.get_or_create_collection(
            name="rostros",
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        print(f"Error al inicializar ChromaDB: {e}")
        return

    # 2. Inicialización de modelos de visión
    detector = inicializar_detector_yunet()
    reconocedor = inicializar_reconocedor_sface()

    if detector is None or reconocedor is None:
        print("Error: No se pudieron inicializar los modelos de visión (YuNet / SFace).")
        return

    cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara en el índice {webcam_index}.")
        return

    capturas = 0
    tiempo_ultima_captura = 0.0

    print(f"Iniciando registro para '{nombre_usuario}' en ChromaDB ({muestras_totales} muestras). Presiona ESC para cancelar.")

    while capturas < muestras_totales and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        tiempo_actual = time.time()

        if faces is not None:
            for face in faces:
                x, y, w_box, h_box = face[0:4].astype(int)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w, x + w_box), min(h, y + h_box)

                # Guardar embedding en ChromaDB si se cumplió el retardo de 0.1s
                if (tiempo_actual - tiempo_ultima_captura) >= retardo_segundos and capturas < muestras_totales:
                    face_align = reconocedor.alignCrop(frame, face)
                    face_feature = reconocedor.feature(face_align)
                    embedding = face_feature.flatten().tolist()

                    id_unico = f"{nombre_usuario}_{uuid.uuid4()}"
                    coleccion.add(
                        ids=[id_unico],
                        embeddings=[embedding],
                        metadatas=[{"usuario": nombre_usuario}]
                    )
                    capturas += 1
                    tiempo_ultima_captura = tiempo_actual

                # Feedback visual en cada frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.putText(
            frame,
            f"Capturas: {capturas}/{muestras_totales}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )
        cv2.imshow("T02 - Registro de Rostro con ChromaDB", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Registro completado: {capturas} embeddings guardados en ChromaDB ('rostros') para '{nombre_usuario}'.")


if __name__ == "__main__":
    nombre = input("Ingresa el nombre de la persona a registrar: ")
    if nombre.strip():
        registrar_usuario(nombre.strip())
    print(f"Registro completado: {capturas} embeddings guardados en ChromaDB ('rostros') para '{nombre_usuario}'.")


if __name__ == "__main__":
    nombre = input("Ingresa el nombre de la persona a registrar: ")
    if nombre.strip():
        registrar_usuario(nombre.strip())