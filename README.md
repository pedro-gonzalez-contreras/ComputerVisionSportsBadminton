# Computer Vision Sports Badminton
Herramienta observacional para análisis de partidos deportivos completamente automatizada gracias a herramientas de Computer Vison, Machine Learning y Deep Learning. El sistema ha sido entrenado específicamente para partidos de Badminton.

- El sistema es capaz de identificar los frames en donde el partido está en juego, separandolos de repeticiones, calentamientos previos, descansos entre punto y punto, etc. Modelos VGG16 y k-Nearest Neighbors (KNN). Accuracy 95% Recall Clase Juego 99%.

- Posteriormente, gracias a un modelo Yolo v3 pre-entrenado se extraen las Bounding Boxes de cada jugador en cada frame y se determina su posición en la pista con una homografía. Posibilidad de crear mapas de calor, transciones habituales del jugador, etc.

- Además, con un modelo HRNet se extraen 16 keypoints del esqueleto de cada jugador. Permite estudio de la técnica de cada juagdor, posturas, identificación de golpeo, fatiga del jugador, etc.

- A continuación, con clusterizaciones K-Means y herramientas de OpenCV se crea un modelo para identificar qué jugador está en la pista inferior, cual en la superior y cuando se producen cambios de pista, gracias al color de su vestimenta.

- Por último, con los keypoints de los jugadores y las HOG features de sus imágenes se ha intentado generar un clasificador que determine el golpeo que está realizando cada jugador en cada frame. En proceso

## Mejoras al proyecto:
- Etiquetar imágenes propias para identificar las esquinas de la pista (Automatizar homografía) y otros elementos de la imagen (Postes,árbitro,etc). Dificultad: Fácil.
- Etiquetar BoundingBoxes propias para mejorar precision del modelo de Object Detection con dos clases: Bottom Player y Top Player. Dificultad: Fácil.
- Obtener la puntuación del partido. Usar imágenes y keypoints de los primeros frames de cada punto para identificar jugador que saca y ganador del punto anterior. Dificultad: Media.
- Identificar y hacer tracking del volante para situar momento del golpeo y estudiar trayectorias. Dificultad: Difícil.
