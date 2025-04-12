import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

# Configuración de la clase para objetos interactivos
@dataclass
class VirtualObject:
    image: np.ndarray
    x: int
    y: int
    scale: float = 1.0
    
    def get_scaled_size(self) -> Tuple[int, int]:
        h, w = self.image.shape[:2]
        return int(w * self.scale), int(h * self.scale)
    
    def draw(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        obj_w, obj_h = self.get_scaled_size()
        
        # Calcular posición para centrar la imagen
        x1 = max(0, min(self.x - obj_w // 2, w - obj_w))
        y1 = max(0, min(self.y - obj_h // 2, h - obj_h))
        
        # Redimensionar imagen
        resized = cv2.resize(self.image, (obj_w, obj_h), interpolation=cv2.INTER_AREA)
        
        # Crear máscara para transparencia si la imagen tiene canal alfa
        if resized.shape[2] == 4:
            mask = resized[:, :, 3] / 255.0
            inv_mask = 1.0 - mask
            
            # Región de interés en el frame
            roi = frame[y1:y1+obj_h, x1:x1+obj_w]
            
            # Superponer imagen con transparencia
            for c in range(3):
                roi[:, :, c] = (mask * resized[:, :, c] + inv_mask * roi[:, :, c]).astype(np.uint8)
        else:
            frame[y1:y1+obj_h, x1:x1+obj_w] = resized

# Inicialización de MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Cargar imagen del objeto (asegúrate de tener una imagen PNG con transparencia)
try:
    obj_image = cv2.imread("object.png", cv2.IMREAD_UNCHANGED)
    if obj_image is None:
        raise FileNotFoundError("No se encontró object.png")
except Exception as e:
    print(f"Error al cargar la imagen: {e}")
    # Fallback a un rectángulo si falla la carga
    obj_image = np.zeros((100, 100, 4), dtype=np.uint8)
    obj_image[:, :, :3] = (0, 255, 0)  # Verde
    obj_image[:, :, 3] = 255  # Opaco

# Crear objeto virtual
virtual_obj = VirtualObject(
    image=obj_image,
    x=200,
    y=200,
    scale=0.5
)

# Captura de video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

holding = False
smooth_factor = 0.3  # Para suavizar el movimiento

def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return np.linalg.norm(np.array(p1) - np.array(p2))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Procesar detección de manos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            
            # Puntos clave
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Convertir a píxeles
            x_index = int(index_tip.x * w)
            y_index = int(index_tip.y * h)
            x_thumb = int(thumb_tip.x * w)
            y_thumb = int(thumb_tip.y * h)
            
            # Dibujar mano con estilo personalizado
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2),
                mp_draw.DrawingSpec(color=(255, 0, 255), thickness=4)
            )
            
            # Detectar agarre
            distance = calculate_distance((x_index, y_index), (x_thumb, y_thumb))
            holding = distance < 40
            
            # Efecto visual al agarrar
            if holding:
                cv2.circle(frame, (x_index, y_index), 15, (0, 255, 0), 2)
                virtual_obj.scale = min(virtual_obj.scale + 0.02, 0.7)  # Animación de escala
            else:
                virtual_obj.scale = max(virtual_obj.scale - 0.02, 0.5)
            
            # Mover objeto con suavizado
            if holding:
                target_x, target_y = x_index, y_index
                virtual_obj.x = int(virtual_obj.x * (1 - smooth_factor) + target_x * smooth_factor)
                virtual_obj.y = int(virtual_obj.y * (1 - smooth_factor) + target_y * smooth_factor)

    # Dibujar objeto
    virtual_obj.draw(frame)
    
    # Añadir texto informativo
    cv2.putText(frame, "Presiona 'q' para salir", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Drag & Drop Virtual", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()