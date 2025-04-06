import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Captura de video
cap = cv2.VideoCapture(0)
obj_x, obj_y = 200, 200  # Posición inicial del objeto
obj_size = 50
holding = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Espejo para mejor experiencia
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extraer puntos clave
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convertir coordenadas a píxeles
            h, w, _ = frame.shape
            x_index, y_index = int(index_finger.x * w), int(index_finger.y * h)
            x_thumb, y_thumb = int(thumb.x * w), int(thumb.y * h)

            # Dibujar la mano
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calcular distancia entre índice y pulgar
            distance = np.linalg.norm(np.array([x_index, y_index]) - np.array([x_thumb, y_thumb]))

            # Detectar gesto de agarre
            if distance < 30:
                holding = True
            else:
                holding = False

            # Si está agarrando, mover el objeto
            if holding:
                obj_x, obj_y = x_index, y_index

    # Dibujar objeto
    cv2.rectangle(frame, (obj_x - obj_size//2, obj_y - obj_size//2), 
                  (obj_x + obj_size//2, obj_y + obj_size//2), (0, 255, 0), -1)

    cv2.imshow("Drag & Drop Virtual", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
