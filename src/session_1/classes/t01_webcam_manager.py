# Task T01
# - T01: Extraer de una Webcam, el streaming de video y mostrarlo en una ventana 
# Edgar Daniel Martínez López - 16/09/2026

import cv2
from pygrabber.dshow_graph import FilterGraph

class WebcamManager:
    FLIP_MODES = {
        'normal': None,
        'y-axis': 1,      
        'x-axis': 0,      
        'both-axis': -1   
    }
    """Class to show input devices and show webcam streaming."""

    @staticmethod
    def show_devices():
        """Method to show all the webcam devices detected in the computer"""
        graph = FilterGraph()
        devices = graph.get_input_devices()

        if not devices:
            print("Not devices were found!")
            return []

        for index, name in enumerate(devices):
            print(f"Index [{index}] -> {name}")
            
        return devices

    def play_streaming(self, index=0, invert='normal', window_name='Webcam Video'):
        """
        This method loads the streaming video from the i-th input device.
        Also, allows inverting horitontally the image through invert parameter
        """
        
        if invert not in self.FLIP_MODES:
            print("Error: The given invert value is not accepted")
        
        flip_code = self.FLIP_MODES[invert]
        # Este index permite identificar a que dispositivo nos estamos conectando
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print(f"Error: The device with index: {index} was not found.")
            return

        while True:
            # Este método combina 2 operaciones cap.grap() y cap.retrieve()
            # cap.grab() obtiene la informacion obtiene el siguiente fotograma al input device 
            # cap.retrieve() decodifica el fotograma a una matriz de Numpy
            # devolviendo 2 valores
            # 1. ret: return value (boolean -> para saber si s obtuvo el fotograma
            # 2. img (np.array): la imagen obtenida o None
            ret, img = cap.read()
            
            if ret:
                # Aplica el flip indicado
                if flip_code is not None:
                    img = cv2.flip(img, flip_code)

                # Mostrar el fotograma
                cv2.imshow(window_name, img)
                
                # Máscara AND de '11111111' para obtener valor dentro de rango ASCII
                k = cv2.waitKey(1) & 0xFF
                
                # 27 corresponde a la tecla ESC en ASCIII
                if k == 27:
                    break
            else:
                break      

        # Liberar input device
        cap.release()
        cv2.destroyAllWindows()
        