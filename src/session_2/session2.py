# Archivo Principal Sesión 2 - Embodied AI con ChromaDB y Telemetría
# Edgar Daniel Martínez López - Octubre 2026

import cv2
from t01_deteccion_rostros import inicializar_detector_yunet, detectar_rostros
from t02_registrar_rostro import registrar_usuario
from t03_reconocimiento_facial import reconocer_en_tiempo_real
from t04_deteccion_emociones import detectar_emociones_tiempo_real


def probar_t01_deteccion(webcam_index=0):
    """Prueba de detección de rostros con YuNet."""
    detector = inicializar_detector_yunet()
    if detector is None:
        return

    cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara con índice {webcam_index}.")
        return

    print("\n--- Iniciando T01: Detección de Rostros con YuNet. Presiona ESC para salir ---")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_dibujado, _ = detectar_rostros(frame, detector)
        cv2.imshow("T01 - Detección de Rostros (YuNet)", frame_dibujado)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def probar_t03_con_telemetria(webcam_index=0):
    """Función de compatibilidad para invocar el reconocimiento facial con telemetría y ChromaDB."""
    reconocer_en_tiempo_real(webcam_index=webcam_index)


if __name__ == "__main__":
    index_camara = 0

    while True:
        print("\n" + "=" * 45)
        print(" MENÚ SESIÓN 2 - EMBODIED AI (CHROMADB) ".center(45, "="))
        print("=" * 45)
        print("1. T01: Detección de Rostros (YuNet)")
        print("2. T02: Registrar Nuevo Rostro (ChromaDB)")
        print("3. T03: Reconocimiento con TELEMETRÍA (ChromaDB)")
        print("4. T04: Detección de Emociones (Ollama)")
        print("5. Salir")
        print("=" * 45)

        opcion = input("Selecciona una opción (1-5): ").strip()

        if opcion == '1':
            probar_t01_deteccion(webcam_index=index_camara)
        elif opcion == '2':
            nombre = input("Ingresa el nombre de la persona a registrar: ").strip()
            if nombre:
                registrar_usuario(nombre_usuario=nombre, muestras_totales=150, webcam_index=index_camara, retardo_segundos=0.1)
            else:
                print("El nombre no puede estar vacío.")
        elif opcion == '3':
            reconocer_en_tiempo_real(webcam_index=index_camara)
        elif opcion == '4':
            detectar_emociones_tiempo_real(webcam_index=index_camara)
        elif opcion == '5':
            print("Cerrando sesión de Embodied AI...")
            break
        else:
            print("Opción no válida. Por favor selecciona del 1 al 5.")
