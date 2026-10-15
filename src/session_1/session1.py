from classes.t01_webcam_manager import WebcamManager
from classes.t02_object_detection import ObjectDetection
from classes.t03_face_detection import FaceDetection

if __name__ == "__main__":
    # --------------- Task T01 ---------------
    # cam = WebcamManager()
    # # 1. Buscar los dispoitivos conectados
    # print("--- Dispositivos detectados ---")
    # cam.show_devices()
    # print("--------------------------------\n")
    # # 2. Cargar el streaming de la webcam [index]:
    # cam.play_streaming(index=1, invert='normal')
    
    
    # --------------- Task T02 ---------------
    obj = ObjectDetection()
    image_path = "src/session_1/assets/images/dog.jpeg"
    video_path = "src/session_1/assets/videos/people.mp4"
    conf = 0.45
    
    # 1. Detección de objetos en imágenes
    obj.object_detection_image(image_path=image_path, conf=conf)
    
   # 2. Detección de objetos en video
    obj.object_detection_video(source='webcam', webcam_index=1, conf=conf)

    # --------------- Task T03 ---------------
    # faceDet = FaceDetection()
    # image_path = "src/session_1/assets/images/face01.jpg"
    # video_path = "src/session_1/assets/videos/people.mp4"
    
    # 1. Detección de rostros en imágenes
    # faceDet.detect_image(image_path=image_path)
    
    # 2. Detección de rostos en vídeos
    # faceDet.detect_video(source='video', video_path=video_path)
    
    
    