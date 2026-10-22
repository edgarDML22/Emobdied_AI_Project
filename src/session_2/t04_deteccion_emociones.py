# Tarea T04: Detección de emociones vía Webcam integrando Ollama (Moondream/Llava)
# Edgar Daniel Martínez López - Octubre 2026

import cv2
import threading
import time
import base64
import requests
import json
import re
from t01_deteccion_rostros import inicializar_detector_yunet, detectar_rostros

# Configuración Global
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODELO_OLLAMA = "moondream"  # Modelo activo: 'moondream' (1.8B) o 'llava' (7B)
INTERVALO_ANALISIS = 3.0     # Segundos entre análisis automáticos (si está activo el modo auto)

# Paleta de colores en formato BGR para cada emoción
COLORES_EMOCIONES = {
    "Feliz": (46, 204, 113),         # Verde esmeralda
    "Triste": (220, 120, 30),        # Azul
    "Enojado": (30, 30, 230),        # Rojo intenso
    "Sorprendido": (180, 50, 180),   # Magenta / Púrpura
    "Miedo": (0, 140, 255),          # Ámbar / Naranja
    "Neutral": (160, 140, 110),      # Pizarra / Gris neutro
    "Detectando...": (200, 150, 30), # Azul acero
    "Desconocido": (120, 120, 120)   # Gris oscuro
}

# Variables globales para el hilo asíncrono
emocion_actual = "Neutral"
descripcion_actual = ""
analisis_en_progreso = False
tiempo_ultimo_analisis = 0.0
tiempo_inicio_peticion = 0.0
estado_analisis = "Presiona 'E' para analizar"


def codificar_imagen_base64(imagen):
    """
    Redimensiona el recorte facial a 224x224 y lo convierte a JPEG en base64.
    Esto aligera la carga de red y optimiza la inferencia del encoder de visión.
    """
    img_redim = cv2.resize(imagen, (224, 224), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', img_redim, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def extraer_emocion_de_respuesta(texto_ollama):
    """
    Analiza la descripción anatómica (boca, cejas, ojos) generada por el VLM.
    Sin sesgos de selección múltiple ni falsos positivos hacia 'Feliz'.
    """
    t = texto_ollama.lower().strip()

    if len(t) < 5 or t.startswith('!'):
        return 'Neutral'

    # 1. Comprobar si no hay sonrisa o el rostro está neutro/cerrado
    no_sonrie = any(k in t for k in [
        'not smiling', 'no smile', 'without a smile', 'does not smile', 
        "doesn't smile", 'unsmiling', 'straight mouth', 'closed mouth', 
        'thin line for a mouth', 'thin black line', 'straight line', 'neutral expression'
    ])

    # 2. Feliz: Se detecta explícitamente una sonrisa o gesto alegre
    tiene_sonrisa = any(k in t for k in [
        'smiling', 'smile', 'smiles', 'grin', 'grinning', 
        'laughing', 'cheerful', 'smiley', 'curved upwards', 'happy'
    ])
    if tiene_sonrisa and not no_sonrie:
        return 'Feliz'

    # 3. Enojado: Ceño fruncido, mirada fija, hostil o enfado
    if any(k in t for k in [
        'furrow', 'squinting', 'glare', 'glaring', 'scowl', 
        'angry', 'anger', 'furious', 'rage', ' mad ', 'mad,', 'mad.', 'hostil', 'stern'
    ]):
        return 'Enojado'

    # 4. Triste: Boca hacia abajo, desánimo, tristeza o llanto
    if any(k in t for k in [
        'sad', 'sadness', 'downturn', 'downward', 'droop', 
        'disappoint', 'unhappy', 'pout', 'pouting', 'cry', 
        'crying', 'tears', 'grief', 'melanchol', 'downcast'
    ]):
        return 'Triste'
    if 'frown' in t and not tiene_sonrisa:
        return 'Triste'

    # 5. Sorprendido / Miedo
    if any(k in t for k in ['scared', 'afraid', 'terrifi', 'panic', 'fear']):
        return 'Miedo'
    if any(k in t for k in ['surpris', 'shock', 'wide open', 'gasp', 'open-mouthed']):
        return 'Sorprendido'

    # 6. Neutral: Por defecto ante rostros serios, descansados o inexpresivos
    return 'Neutral'


def solicitar_emocion_a_ollama(imagen_recortada):
    """Ejecuta la petición a Ollama en un hilo separado para no congelar la cámara."""
    global emocion_actual, descripcion_actual, analisis_en_progreso, estado_analisis

    try:
        img_b64 = codificar_imagen_base64(imagen_recortada)

        # Prompt descriptivo neutral sin nombres de emociones para eliminar el sesgo de Moondream
        prompt = "Describe the facial expression in this photo. Detail the mouth, eyes, and eyebrows."

        # Ajuste dinámico de timeout y tokens según modelo
        if MODELO_OLLAMA == "llava":
            timeout_seg = 120  # LLaVA 7B en CPU requiere de 45 a 70 segundos
            num_tokens = 35
        else:
            timeout_seg = 45   # Moondream 1.8B requiere de 10 a 16 segundos
            num_tokens = 40

        payload = {
            "model": MODELO_OLLAMA,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "num_predict": num_tokens,
                "temperature": 0.1,
                "repeat_penalty": 1.15
            }
        }

        respuesta = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout_seg)
        if respuesta.status_code == 200:
            datos = respuesta.json()
            raw_response = datos.get("response", "").strip()
            descripcion_actual = raw_response.replace('\n', ' ')
            emocion_detectada = extraer_emocion_de_respuesta(raw_response)
            emocion_actual = emocion_detectada
            estado_analisis = f"Detectado: {emocion_actual}"
            print(f"\n[OLLAMA - {MODELO_OLLAMA.upper()}] Emocion clasificada: {emocion_actual}")
            print(f"[OLLAMA] Descripcion visual: \"{descripcion_actual}\"")
        else:
            estado_analisis = f"Error Ollama ({respuesta.status_code})"
            print(f"\n[OLLAMA] Error HTTP {respuesta.status_code}")
    except requests.exceptions.Timeout:
        estado_analisis = f"Tiempo agotado en {MODELO_OLLAMA.upper()}"
        print(f"\n[OLLAMA] Tiempo de espera agotado ({timeout_seg}s). En CPU el modelo tardó demasiado.")
    except requests.exceptions.ConnectionError:
        estado_analisis = "Sin conexion (inicia Ollama)"
        print("\n[OLLAMA] No se pudo conectar a Ollama. Asegurate de que este corriendo.")
    except Exception as e:
        estado_analisis = "Error en consulta"
        print(f"\n[OLLAMA] Excepcion: {e}")
    finally:
        analisis_en_progreso = False


def dibujar_etiqueta_legible(frame, texto, x, y, color_fondo, color_texto=(255, 255, 255)):
    """
    Dibuja una placa o badge con fondo de color y texto blanco con borde oscuro.
    Garantiza máxima legibilidad incluso sobre fondos completamente blancos o muy iluminados.
    """
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.65
    grosor = 2

    (ancho_t, alto_t), baseline = cv2.getTextSize(texto, fuente, escala, grosor)

    # Posición del badge sobre la caja del rostro
    y_sup = max(0, y - alto_t - 14)
    y_inf = y_sup + alto_t + 12
    x_der = min(frame.shape[1], x + ancho_t + 16)

    # Rectángulo con el color de la emoción
    cv2.rectangle(frame, (x, y_sup), (x_der, y_inf), color_fondo, -1)
    # Borde exterior oscuro
    cv2.rectangle(frame, (x, y_sup), (x_der, y_inf), (30, 30, 30), 1)

    # Contorno de texto negro + relleno blanco para alto contraste
    pos_texto = (x + 8, y_inf - 6)
    cv2.putText(frame, texto, pos_texto, fuente, escala, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(frame, texto, pos_texto, fuente, escala, color_texto, grosor, cv2.LINE_AA)


def detectar_emociones_tiempo_real(webcam_index=0):
    global analisis_en_progreso, tiempo_ultimo_analisis, tiempo_inicio_peticion
    global emocion_actual, descripcion_actual, estado_analisis, MODELO_OLLAMA

    # Reiniciar variables de estado al iniciar cada sesión
    analisis_en_progreso = False
    emocion_actual = "Neutral"
    descripcion_actual = ""
    estado_analisis = "Presiona 'E' para capturar"
    tiempo_ultimo_analisis = 0.0
    tiempo_inicio_peticion = 0.0

    modo_automatico = False  # Inicia en modo manual para control preciso con tecla 'E'
    mensaje_notificacion = "Presiona 'E' para enviar fotograma a Ollama"
    tiempo_notificacion = time.time() + 3.0

    # Inicializar detector YuNet importado de T01
    detector = inicializar_detector_yunet()
    if detector is None:
        return

    cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara con índice {webcam_index}.")
        return

    print(f"\n{'='*60}")
    print(f"--- T04: DETECCIÓN DE EMOCIONES CON {MODELO_OLLAMA.upper()} ---")
    print(f"{'='*60}")
    print("CONTROLES:")
    print("  [E]   -> Capturar el fotograma actual y enviarlo a Ollama")
    print("  [M]   -> Alternar modelo (moondream ~12s | llava ~50s en CPU)")
    print("  [A]   -> Alternar modo Automatico / Manual")
    print("  [ESC] -> Salir al menu principal")
    print(f"{'='*60}\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        detector.setInputSize((w, h))
        _, faces = detectar_rostros(frame, detector)

        # Color asociado a la emoción detectada
        color_emocion = COLORES_EMOCIONES.get(emocion_actual, COLORES_EMOCIONES["Neutral"])

        ultimo_rostro_recortado = None

        if faces is not None and len(faces) > 0:
            face = faces[0]
            x, y, w_box, h_box = face[0:4].astype(int)

            # Limitar coordenadas dentro de los márgenes del frame
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + w_box), min(h, y + h_box)

            # Dibujar la caja delimitadora con el color de la emoción
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_emocion, 2)

            rostro_crop = frame[y1:y2, x1:x2]
            if rostro_crop.size > 0:
                ultimo_rostro_recortado = rostro_crop.copy()

            # Texto legible sobre el rostro
            texto_etiqueta = f"Emocion: {emocion_actual}"
            dibujar_etiqueta_legible(frame, texto_etiqueta, x1, y1, color_emocion)

            # Si el modo automático está activo, analiza periódicamente
            tiempo_actual = time.time()
            if modo_automatico and not analisis_en_progreso and (tiempo_actual - tiempo_ultimo_analisis) >= INTERVALO_ANALISIS:
                analisis_en_progreso = True
                tiempo_inicio_peticion = tiempo_actual
                tiempo_ultimo_analisis = tiempo_actual
                estado_analisis = f"Analizando auto ({MODELO_OLLAMA})..."

                hilo = threading.Thread(target=solicitar_emocion_a_ollama, args=(ultimo_rostro_recortado,))
                hilo.daemon = True
                hilo.start()

        # Construir texto de la barra inferior
        tiempo_actual = time.time()
        if analisis_en_progreso:
            tiempo_transcurrido = int(tiempo_actual - tiempo_inicio_peticion)
            texto_inferior = f"Consultando Ollama ({MODELO_OLLAMA.upper()})... {tiempo_transcurrido}s en CPU"
            color_barra = (0, 165, 255)
        else:
            modo_str = "AUTO" if modo_automatico else "MANUAL [Presiona 'E']"
            texto_inferior = f"Modelo: {MODELO_OLLAMA.upper()} | Modo: {modo_str} | Estado: {estado_analisis}"
            color_barra = (0, 220, 0)

        # Barra inferior de estado (fondo oscuro de alto contraste)
        cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, texto_inferior, (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color_barra, 1, cv2.LINE_AA)

        # Barra superior con instrucciones y avisos temporales
        cv2.rectangle(frame, (0, 0), (w, 30), (30, 30, 30), -1)
        if tiempo_actual < tiempo_notificacion:
            cv2.putText(frame, mensaje_notificacion, (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            estimado = "~12s" if MODELO_OLLAMA == "moondream" else "~50s"
            instruccion = f"[E] Capturar | [M] Modelo: {MODELO_OLLAMA.upper()} ({estimado}) | [A] Auto | [ESC] Salir"
            cv2.putText(frame, instruccion, (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("T04 - Deteccion de Emociones (Ollama)", frame)

        # Manejo de teclas del usuario
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC: Salir
            break
        elif key in (ord('e'), ord('E')):  # Tecla 'E': Captura instantánea
            if analisis_en_progreso:
                mensaje_notificacion = "Ollama ocupado, espera a que termine..."
                tiempo_notificacion = time.time() + 2.0
                print("\n[AVISO] Ollama aún está analizando la solicitud anterior...")
            elif ultimo_rostro_recortado is None:
                mensaje_notificacion = "No se detecta rostro. Centrate en la camara."
                tiempo_notificacion = time.time() + 2.0
                print("\n[AVISO] No se detecta ningún rostro para capturar.")
            else:
                analisis_en_progreso = True
                tiempo_inicio_peticion = time.time()
                tiempo_ultimo_analisis = time.time()
                estado_analisis = f"Analizando con {MODELO_OLLAMA.upper()}..."
                mensaje_notificacion = f"Capturado! Enviando a {MODELO_OLLAMA.upper()}..."
                tiempo_notificacion = time.time() + 2.5
                print(f"\n[OK] ¡Fotograma capturado! Enviando a {MODELO_OLLAMA.upper()}...")

                hilo = threading.Thread(target=solicitar_emocion_a_ollama, args=(ultimo_rostro_recortado,))
                hilo.daemon = True
                hilo.start()
        elif key in (ord('m'), ord('M')):  # Tecla 'M': Alternar modelo
            if analisis_en_progreso:
                mensaje_notificacion = "Espera a que termine el analisis para cambiar modelo"
                tiempo_notificacion = time.time() + 2.0
            else:
                MODELO_OLLAMA = "llava" if MODELO_OLLAMA == "moondream" else "moondream"
                est = "~50s en CPU" if MODELO_OLLAMA == "llava" else "~12s en CPU"
                mensaje_notificacion = f"Modelo: {MODELO_OLLAMA.upper()} ({est})"
                tiempo_notificacion = time.time() + 3.0
                print(f"\n[MODELO] Cambiado a: {MODELO_OLLAMA.upper()} ({est})")
        elif key in (ord('a'), ord('A')):  # Tecla 'A': Alternar modo Auto/Manual
            modo_automatico = not modo_automatico
            estado_str = "ACTIVADO" if modo_automatico else "DESACTIVADO (Manual con 'E')"
            mensaje_notificacion = f"Modo Automatico: {estado_str}"
            tiempo_notificacion = time.time() + 2.5
            print(f"\n[MODO] Modo Automatico {estado_str}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detectar_emociones_tiempo_real()

    detectar_emociones_tiempo_real()