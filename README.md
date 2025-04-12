🎯 Descripción General:

Desarrollar un sistema interactivo de “drag & drop” (arrastrar y soltar) utilizando Python, implementando el seguimiento de manos con la biblioteca OpenCV y MediaPipe. El sistema detectará el movimiento de la mano en tiempo real a través de la cámara web, identificando cuándo el usuario quiere agarrar, mover y soltar objetos virtuales mediante gestos naturales.

🔧 Tecnologías a utilizar:

Python 3.x
OpenCV para captura de video y procesamiento de imagen
MediaPipe para detección y seguimiento de manos
Tkinter o interfaz gráfica simple (opcional)
Git + GitHub para control de versiones
📋 Objetivos del Proyecto:

Implementar un sistema de seguimiento de manos en tiempo real.
Detectar gestos de “agarre” (como juntar dedos índice y pulgar).
Permitir que el usuario mueva un objeto virtual con su mano.
Soltar el objeto con otro gesto (como separar los dedos).
Visualizar los objetos y gestos en pantalla.
Versionar el proyecto correctamente en GitHub.
Generar un video demostrativo del funcionamiento del proyecto.


------------------------------------------------------------------------------------------------------------
1. Importaciones
      import cv2
      import mediapipe as mp
      import numpy as np
      from dataclasses import dataclass
      from typing import Tuple, Optional
cv2: OpenCV para captura de video, procesamiento de imágenes y renderizado.
mediapipe: Biblioteca para detección de manos (mp.solutions.hands) y dibujo de puntos clave (mp.solutions.drawing_utils).
numpy: Para cálculos matemáticos, como distancias entre puntos.
dataclasses: Simplifica la creación de la clase VirtualObject.
typing: Mejora la legibilidad con anotaciones de tipo (Tuple, Optional).

2. Definición de la Clase VirtualObject
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
        
        #Calcular posición para centrar la imagen
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
   Define una clase para representar objetos virtuales que el usuario puede mover.
Atributos:
image: Imagen del objeto (un array de NumPy, normalmente PNG con canal alfa).
x, y: Coordenadas del centro del objeto.
scale: Escala de la imagen (1.0 = tamaño original).
Métodos:
get_scaled_size: Calcula el ancho y alto de la imagen escalada.
draw: Renderiza la imagen en el frame, manejando transparencia y límites de pantalla.
Cómo se creó:

Se usó @dataclass para simplificar la definición de la clase.
El método draw implementa renderizado con transparencia, crucial para que el objeto (como un ícono o sprite) se vea profesional.
Se añadieron restricciones (max, min) para evitar que el objeto salga de la pantalla.
La lógica de transparencia usa el canal alfa (si existe) para combinar la imagen con el fondo.

3. Inicialización de MediaPipe
       mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

Qué hace:

Configura MediaPipe para detectar una sola mano.
Establece umbrales altos (0.8) para detección y seguimiento, garantizando precisión.
Se limitó a una mano (max_num_hands=1) para simplificar la interacción.
Los umbrales se ajustaron tras pruebas para equilibrar sensibilidad y estabilida

4. Carga de la Imagen del Objeto
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

   Intenta cargar object.png con soporte para canal alfa (IMREAD_UNCHANGED).
Si falla, crea una imagen de respaldo (cuadrado verde opaco).
Cómo se creó:

Se usó un bloque try-except para manejar errores (archivo no encontrado, formato inválido).
El fallback asegura que la aplicación no se bloquee si falta la imagen.

6. Configuración de la Captura de Video
   cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()
Inicia la captura de video desde la cámara predeterminada (índice 0).
Verifica si la cámara está disponible; si no, termina el programa.
Cómo se creó:

Se añadió manejo de errores para evitar fallos si la cámara no funciona.
El índice 0 es estándar para la cámara web integrada.

7. Variables de Estado
   holding = False
smooth_factor = 0.3  # Para suavizar el movimiento


holding: Indica si el usuario está "agarrando" el objeto (pellizco activo).
smooth_factor: Controla la suavidad del movimiento (0.3 = 30% hacia la nueva posición por frame).
Cómo se creó:

smooth_factor se ajustó experimentalmente para un movimiento fluido sin retraso notable.
holding es un simple booleano para rastrear el estado del gesto.

8. Función Auxiliar
   def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return np.linalg.norm(np.array(p1) - np.array(p2))

   
Calcula la distancia euclidiana entre dos puntos (usada para detectar el pellizco).
Cómo se creó:

Se extrajo a una función para reutilización y claridad.
Usa NumPy para un cálculo eficiente.

9. Bucle Principal
    while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

   Lee frames de la cámara en un bucle continuo.
Si no se puede leer un frame, sale del bucle.
Cómo se creó:

Estructura estándar para procesamiento de video en tiempo real con OpenCV.

10. Preprocesamiento del Frame
    frame = cv2.flip(frame, 1)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(rgb_frame)

nvierte el frame horizontalmente (espejo) para una experiencia más natural.
Convierte el frame de BGR (formato de OpenCV) a RGB (requerido por MediaPipe).
Procesa el frame con MediaPipe para detectar manos.
Cómo se creó:

El flip es un estándar en aplicaciones de seguimiento para que los movimientos sean intuitivos.
La conversión de color es necesaria porque MediaPipe espera RGB.

11. Procesamiento de la Detección de Manos
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
    Si se detectan manos, itera sobre las manos encontradas (aunque está limitado a una).
Obtiene las coordenadas normalizadas (0 a 1) de la punta del índice y el pulgar.
Convierte estas coordenadas a píxeles usando las dimensiones del frame.
Cómo se creó:

Se seleccionaron INDEX_FINGER_TIP y THUMB_TIP porque son ideales para detectar un pellizco.
La conversión a píxeles es directa: multiplica las coordenadas normalizadas por el ancho (w) y alto (h).


12. Dibujo de la Mano
    mp_draw.draw_landmarks(
    frame,
    hand_landmarks,
    mp_hands.HAND_CONNECTIONS,
    mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2),
    mp_draw.DrawingSpec(color=(255, 0, 255), thickness=4)
)

Dibuja los puntos clave y conexiones de la mano en el frame.
Usa colores personalizados: cian (0, 255, 255) para conexiones, magenta (255, 0, 255) para puntos.
Cómo se creó:

Se personalizaron los colores y grosores para un look más atractivo.
DrawingSpec permite ajustar el estilo visual de MediaPipe.

13. Detección del Gesto de Agarre
    distance = calculate_distance((x_index, y_index), (x_thumb, y_thumb))
holding = distance < 40

Calcula la distancia entre el índice y el pulgar.
Si es menor a 40 píxeles, activa el estado holding.
Cómo se creó:

El umbral de 40 píxeles se ajustó tras pruebas para detectar pellizcos de forma confiable.
La función calculate_distance asegura un cálculo preciso.

14. Efectos Visuales al Agarrar
    if holding:
    cv2.circle(frame, (x_index, y_index), 15, (0, 255, 0), 2)
    virtual_obj.scale = min(virtual_obj.scale + 0.02, 0.7)
else:
    virtual_obj.scale = max(virtual_obj.scale - 0.02, 0.5)

    Si holding es verdadero:
Dibuja un círculo verde en la punta del índice como retroalimentación visual.
Aumenta la escala del objeto hasta un máximo de 0.7.
Si no, reduce la escala hasta un mínimo de 0.5.
Cómo se creó:

El círculo se añadió para confirmar visualmente el agarre.
La animación de escala (0.02 por frame) crea un efecto dinámico sin ser abrumador.
Los límites (0.5, 0.7) evitan que el objeto sea demasiado pequeño o grande.

15. Movimiento del Objeto
    if holding:
    target_x, target_y = x_index, y_index
    virtual_obj.x = int(virtual_obj.x * (1 - smooth_factor) + target_x * smooth_factor)
    virtual_obj.y = int(virtual_obj.y * (1 - smooth_factor) + target_y * smooth_factor)

Si el usuario está agarrando, actualiza la posición del objeto.
Usa interpolación lineal para suavizar el movimiento: combina la posición actual con la nueva usando smooth_factor.
Cómo se creó:

La interpolación (smooth_factor = 0.3) asegura movimientos fluidos, evitando saltos bruscos.
Las coordenadas objetivo son las del índice, ya que es el punto principal del gesto.

16. Texto Informativo
  cv2.putText(frame, "Presiona 'q' para salir", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

Muestra un mensaje en la esquina superior izquierda para indicar cómo salir.
Cómo se creó:

Se usó una fuente legible (HERSHEY_SIMPLEX) con tamaño y grosor moderados.
El color blanco es visible en la mayoría de fondos.
    
