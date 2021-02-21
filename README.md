# Computer Vision Sports Badminton
Herramienta observacional para análisis de partidos deportivos completamente automatizada gracias a herramientas de Computer Vison, Machine Learning y Deep Learning. El sistema ha sido entrenado específicamente para partidos de Badminton

El sistema es capaz de identificar los frames en donde el partido está en juego, separandolos de repeticiones, calentamientos previos, descansos entre punto y punto, etc. Modelos VGG16 y k-Nearest Neighbors (KNN). Accuracy 95% Recall Clase Juego 99%
Posteriormente, gracias a un modelo Yolo v3 pre-entrenado se extraen las Bounding Boxes de cada jugador en cada frame y se determina su posición en la pista con una hodografía.
Además, con un modelo HRNet se extraen 16 keypoints del esqueleto de cada jugador:

Por último, con clusterizaciones K-Means y herramientas de OpenCV se crea un modelo para identificar qué jugador está en la pista inferior y cual en la superior y cuando se producen el cambio gracias al color de su vestimenta.

Por último, con los keypoints de los jugadores y las HOG features de sus imágenes se ha intentado generar un clasificador que determine el golpeo que está realizando cada jugador en cada frame 
