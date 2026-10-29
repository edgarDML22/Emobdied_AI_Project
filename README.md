# Embodied AI Project - Sesión 2

Proyecto de **Embodied AI & Visión Computacional** enfocado en detección facial ultra ligera, extracción e indexación de embeddings faciales en base de datos vectorial (**ChromaDB**), reconocimiento facial con telemetría de recursos en tiempo real y clasificación de microexpresiones/emociones mediante Modelos de Visión y Lenguaje (**VLM**) locales con **Ollama**.

---

## 📋 Estructura del Proyecto

```text
Embodied_AI_Project/
│
├── requirements.txt                   # Dependencias del proyecto
├── README.md                          # Guía de instalación y uso
├── bd_vectores/                       # Base de datos persistente ChromaDB (generada en T02)
│
├── face_detection_yunet_2023mar.onnx  # Modelo detector (auto-descargado en la 1ra ejecución)
├── face_recognition_sface_2021dec.onnx# Modelo reconocedor (auto-descargado en la 1ra ejecución)
│
└── src/
    ├── session_1/                     # Ejercicios y fundamentos de la Sesión 1
    └── session_2/                     # Códigos de la Sesión 2 (Embodied AI & ChromaDB)
        ├── session2.py                # Menú interactivo principal
        ├── t01_deteccion_rostros.py   # T01: Detección facial con YuNet
        ├── t02_registrar_rostro.py    # T02: Registro de embeddings en ChromaDB
        ├── t03_reconocimiento_facial.py # T03: Reconocimiento con ChromaDB y Telemetría
        ├── t04_deteccion_emociones.py # T04: Clasificación de emociones con Ollama
        └── visualizar_bd.py           # Utilidad para inspeccionar la colección ChromaDB
```

---

## 🚀 Guía de Inicio Rápido (Setup desde Git Clone)

Sigue estos pasos para clonar y ejecutar el proyecto desde cero:

### 1. Clonar el repositorio

```bash
git clone https://github.com/<TU_USUARIO>/Embodied_AI_Project.git
cd Embodied_AI_Project
```

### 2. Crear y activar el entorno virtual (Python 3.10+)

Se recomienda aislar el entorno mediante `venv`:

**En Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**En Linux / macOS (Bash/Zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar librerías y dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Librerías principales:**
> - `opencv-python`: Inferencia DNN de visión y captura de webcam.
> - `chromadb`: Base de datos vectorial persistente para indexación HNSW (distancia coseno).
> - `psutil`: Telemetría del sistema (RAM y consumo de proceso en MB).
> - `requests`: Cliente HTTP para la API local de inferencia de Ollama.
> - `numpy`, `ultralytics`, `pygrabber`.

---

## 🧠 Modelos y Auto-descarga

### Modelos de Visión ONNX (OpenCV Zoo)
**¡No requieres descargar manualmente los archivos `.onnx`!**
Los scripts cuentan con rutinas de auto-descarga:
1. **YuNet (`face_detection_yunet_2023mar.onnx`)**: Detector facial ultra ligero. Si no existe localmente, `t01_deteccion_rostros.py` lo descargará automáticamente desde el repositorio oficial de OpenCV Zoo la primera vez que se ejecute.
2. **SFace (`face_recognition_sface_2021dec.onnx`)**: Extractor de embeddings faciales de 128 dimensiones. `t03_reconocimiento_facial.py` lo descargará automáticamente en su primera ejecución.

### Modelos de Lenguaje y Visión (Ollama VLM) — *Requerido solo para T04*
Para la tarea de **Detección de Emociones (T04)** se utiliza un Modelo de Lenguaje de Visión (VLM) ejecutado en local:

1. Instala Ollama desde [ollama.com](https://ollama.com).
2. Descarga el modelo liviano recomendado (**Moondream 1.8B**, optimizado para CPU):
   ```bash
   ollama pull moondream
   ```
3. *(Opcional)* Si cuentas con tarjeta gráfica dedicada (GPU) y deseas mayor resolución descriptiva, puedes descargar **LLaVA 7B**:
   ```bash
   ollama pull llava
   ```
4. Asegúrate de que el servicio de Ollama esté ejecutándose en segundo plano (`http://localhost:11434`).

---

## 🎮 Ejecución

> [!IMPORTANT]
> Ejecuta siempre los scripts **desde la carpeta raíz** del proyecto (`Embodied_AI_Project/`) para que los modelos `.onnx` y la base de datos vectorial `./bd_vectores/` se ubiquen correctamente.

### Menú Interactivo Principal:
```bash
python src/session_2/session2.py
```

Al iniciar verás el siguiente menú en consola:
```text
=============================================
=== MENÚ SESIÓN 2 - EMBODIED AI (CHROMADB) ===
=============================================
1. T01: Detección de Rostros (YuNet)
2. T02: Registrar Nuevo Rostro (ChromaDB)
3. T03: Reconocimiento con TELEMETRÍA (ChromaDB)
4. T04: Detección de Emociones (Ollama)
5. Salir
=============================================
```

---

## 📌 Descripción de las Tareas (Sesión 2)

### 1. T01: Detección de Rostros (YuNet)
- **Objetivo:** Inferencia de alta velocidad sobre webcam utilizando la arquitectura liviana YuNet en OpenCV DNN.
- **Salida:** Muestra la caja delimitadora (bounding box) y el índice de confianza sobre el rostro detectado. Presiona `ESC` para salir.

### 2. T02: Registrar Nuevo Rostro en ChromaDB
- **Objetivo:** Creación y persistencia de base de datos vectorial local (`./bd_vectores`).
- **Flujo:** 
  1. Solicita el nombre del usuario por consola.
  2. Detecta el rostro con YuNet y lo alinea.
  3. Extrae embeddings de 128 dimensiones con SFace cada 0.1 segundos hasta completar 150 muestras.
  4. Almacena las muestras en la colección `rostros` en ChromaDB con metadata asociada. Presiona `ESC` para cancelar anticipadamente.

### 3. T03: Reconocimiento Facial con Telemetría
- **Objetivo:** Búsqueda del vecino más cercano en ChromaDB en tiempo real con monitoreo de rendimiento.
- **Métricas HUD:**
  - **Identidad & Similitud:** Umbral de distancia coseno calibrado para identificar al usuario o marcarlo como *Desconocido*.
  - **Telemetría discreta:** Muestra en pantalla **FPS**, consumo de memoria **RAM (RSS en MB)** y el **Delta de memoria** desde el arranque.

### 4. T04: Detección de Emociones con Ollama VLM
- **Objetivo:** Inferencia asíncrona mediante un hilo secundario para enviar recortes faciales a Ollama sin congelar la webcam.
- **Controles interactivos de ventana:**
  - **`[E]`**: Captura el frame actual y envía la petición a Ollama.
  - **`[M]`**: Alterna entre el modelo `moondream` (~10-15s en CPU) y `llava` (~50s en CPU).
  - **`[A]`**: Alterna el modo Automático (analiza periódicamente cada 3 segundos) y Manual (tecla `E`).
  - **`[ESC]`**: Vuelve al menú principal.

---

## 🛠️ Resolución de Problemas Frecuentes

- **Error al abrir la cámara (`cv2.VideoCapture`)**:
  - Si tu equipo tiene varias cámaras (ej. cámara integrada y cámara web USB), modifica el índice `webcam_index=0` a `1` en `session2.py` o en la llamada de la función.
  - En Windows se utiliza el backend `cv2.CAP_DSHOW` para evitar bloqueos de captura.
- **Error: "La colección 'rostros' en ChromaDB está vacía"**:
  - Debes registrar al menos un usuario ejecutando primero la opción **T02** antes de iniciar el reconocimiento en **T03**.
- **Error: "Sin conexión (inicia Ollama)" en T04**:
  - Inicia la aplicación de Ollama en tu equipo o ejecuta `ollama serve` en una terminal independiente.
