# Tarea T03: Reconocimiento Facial con ChromaDB y Telemetría Discreta
# Edgar Daniel Martínez López - Octubre 2026

import os
import time
import urllib.request
import cv2
import psutil
import chromadb
from t01_deteccion_rostros import inicializar_detector_yunet


def descargar_modelo_si_no_existe(ruta_modelo, url):
    """Descarga el modelo desde la URL dada si no se encuentra localmente."""
    if not os.path.exists(ruta_modelo):
        print(f"Modelo '{ruta_modelo}' no encontrado.")
        print("Iniciando descarga automática (puede tardar unos segundos)...")
        try:
            urllib.request.urlretrieve(url, ruta_modelo)
            print(f"¡Descarga de '{ruta_modelo}' completada con éxito!")
        except Exception as e:
            print(f"Error al descargar el modelo: {e}")
            return False
    return True


def inicializar_reconocedor_sface(model_path="face_recognition_sface_2021dec.onnx"):
    """Inicializa el modelo extractor de características faciales SFace (128D)"""
    url_sface = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    if not descargar_modelo_si_no_existe(model_path, url_sface):
        return None
    return cv2.FaceRecognizerSF.create(model_path, "")


def obtener_memoria_proceso():
    """Retorna la memoria RAM (RSS) consumida por el proceso actual en MB."""
    proceso = psutil.Process(os.getpid())
    return proceso.memory_info().rss / (1024 * 1024)


def dibujar_texto_con_borde(img, texto, posicion, fuente=cv2.FONT_HERSHEY_SIMPLEX, escala=0.4, color=(0, 255, 255), grosor=1):
    """Dibuja texto con un borde negro fino para que sea legible y discreto sin cajas sólidas."""
    x, y = posicion
    cv2.putText(img, texto, (x, y), fuente, escala, (0, 0, 0), grosor + 1, cv2.LINE_AA)
    cv2.putText(img, texto, (x, y), fuente, escala, color, grosor, cv2.LINE_AA)


def reconocer_en_tiempo_real(path_bd="./bd_vectores", webcam_index=0):
    """
    Realiza reconocimiento facial en tiempo real consultando la base de datos vectorial ChromaDB.
    Incluye telemetría discreta de FPS, RAM y Delta de memoria.
    """
    # 1. Conexión a ChromaDB con manejo de excepciones
    try:
        cliente_chroma = chromadb.PersistentClient(path=path_bd)
        coleccion = cliente_chroma.get_or_create_collection(
            name="rostros",
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        print(f"Error al inicializar ChromaDB: {e}")
        return

    if coleccion.count() == 0:
        print("La colección 'rostros' en ChromaDB está vacía. Registra usuarios con la T02 primero.")
        return

    # 2. Inicialización de modelos de visión
    detector = inicializar_detector_yunet()
    reconocedor = inicializar_reconocedor_sface()

    if detector is None or reconocedor is None:
        print("Error al inicializar los modelos de visión (YuNet / SFace).")
        return

    cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara con índice {webcam_index}.")
        return

    # Umbral de distancia coseno: distancia < 0.637 equivale a similitud > 0.363 (1 - 0.363)
    umbral_distancia = 0.637

    # Telemetría: Registro inicial de memoria
    memoria_base = obtener_memoria_proceso()
    tiempo_ultima_telemetria = 0.0
    fps_display = 0.0
    mem_display = memoria_base
    delta_display = 0.0

    print("Iniciando reconocimiento facial con ChromaDB y Telemetría. Presiona ESC para salir.")

    while cap.isOpened():
        t_inicio_frame = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        if faces is not None:
            for face in faces:
                x, y, w_box, h_box = face[0:4].astype(int)
                x1, y1 = max(0, x), max(0, y)

                # Extraer embedding 128D usando SFace
                face_align = reconocedor.alignCrop(frame, face)
                face_feature = reconocedor.feature(face_align)
                embedding = face_feature.flatten().tolist()

                # Consulta en ChromaDB buscando el vecino más cercano (n_results=1)
                results = coleccion.query(
                    query_embeddings=[embedding],
                    n_results=1
                )

                mejor_match = "Desconocido"
                confianza = 0.0

                if results and results.get("ids") and len(results["ids"][0]) > 0:
                    distancia = results["distances"][0][0]
                    if distancia < umbral_distancia:
                        # Recuperar nombre de usuario desde metadata o desde prefijo del ID
                        if results.get("metadatas") and results["metadatas"][0] and "usuario" in results["metadatas"][0][0]:
                            mejor_match = results["metadatas"][0][0]["usuario"]
                        else:
                            id_encontrado = results["ids"][0][0]
                            mejor_match = id_encontrado.rsplit("_", 1)[0]

                        confianza = 1.0 - distancia

                color = (0, 255, 0) if mejor_match != "Desconocido" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)

                etiqueta = f"{mejor_match} ({confianza:.2f})" if mejor_match != "Desconocido" else "Desconocido"
                cv2.putText(frame, etiqueta, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Cálculo y actualización de telemetría cada 0.5 segundos para evitar parpadeos
        tiempo_frame = time.time() - t_inicio_frame
        t_actual = time.time()

        if (t_actual - tiempo_ultima_telemetria) >= 0.5:
            fps_display = 1.0 / tiempo_frame if tiempo_frame > 0 else 0.0
            mem_display = obtener_memoria_proceso()
            delta_display = mem_display - memoria_base
            tiempo_ultima_telemetria = t_actual

        # HUD discreto en la esquina inferior izquierda
        alto_frame = frame.shape[0]
        texto_telemetria = f"FPS: {fps_display:.1f} | RAM: {mem_display:.1f}MB (Delta: {delta_display:+.1f}MB)"
        dibujar_texto_con_borde(
            frame,
            texto_telemetria,
            (10, alto_frame - 15),
            fuente=cv2.FONT_HERSHEY_SIMPLEX,
            escala=0.4,
            color=(0, 255, 255),
            grosor=1
        )

        cv2.imshow("T03 - Reconocimiento Facial con ChromaDB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    reconocer_en_tiempo_real()


if __name__ == "__main__":
    reconocer_en_tiempo_real()
    cv2.destroyAllWindows()