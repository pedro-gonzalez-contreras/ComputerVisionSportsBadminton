#!/usr/bin/env python
# coding: utf-8

# In[90]:


#Base libraries
import os
import time
import datetime
import json

# Scientific libraries
import numpy as np
import pandas as pd
from sklearn import svm # or any other function
#import plotly

# Visual libraries
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
# most definitely plotly

# Helper libraries
from tqdm.notebook import tqdm, trange
from colorama import Fore, Back, Style
import warnings
warnings.filterwarnings('ignore')


# Pandas options
pd.set_option('max_colwidth', 40)
pd.options.display.max_columns = None  # Possible to limit
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# Seed value for numpy.random
import random
np.random.seed(42)
from matplotlib.transforms import Bbox
import cv2
path = os.getcwd()
path


# In[91]:


import numpy as np
puntosimagen = []
puntosimagen.append((217.5,684))
puntosimagen.append((438.5,374))
puntosimagen.append((1074,684))
puntosimagen.append((841.0,374))

puntosimagen

esquinas_pista_reales = [(0, 0),(0, 1339),(609, 0), (609, 1339)]
esquinas_pista_reales

def homografia(puntosimagen):
  esquinas_pista_reales = [(0, 0),(0, 1339),(609, 0), (609, 1339)]
  #Hacer homografía
  src_pts = np.array(esquinas_pista_reales, np.float32)
  dst_pts = np.array(puntosimagen, np.float32)
  # TODO ordenar puntos en el mismo orden.
  M, mask = cv2.findHomography(src_pts, dst_pts)
  inv_M = np.linalg.pinv(M)
  return inv_M
  #puntosenpista = cv2.perspectiveTransform(pts, inv_trans)
  #pts = np.array(puntos, np.float32).reshape(-1, 1, 2)
  #dst = cv2.perspectiveTransform(pts, M)
  # Reverse transform
  



# In[92]:


def cajaspista():
    yvalues = np.linspace(0,1339,7)
    xvalues = [0, 175.5, 305 , 434.5, 609]

    xx, yy = np.meshgrid(xvalues, yvalues)

    plt.plot(xx, yy, marker='.', color='k', linestyle='none')

    cajas = []
    for s in range(len(yvalues)-1):
      for t in reversed(range(len(xvalues)-1)):
        caja1 = Bbox([[xx[s,t],yy[s,t]],[xx[s+1,t+1],yy[s+1,t+1]]])
        cajas.append(caja1)

    return cajas


# In[2]:


def incluirzonapista(puntosimagen,nombrearchivocsv,nombrevideo):

    data = pd.read_csv(nombrearchivocsv) 
    df = pd.DataFrame(data)
    df['p0_pto_identif_x'] = (df['p0xmax']+df['p0xmin'])/2
    df['p0_pto_identif_y'] = df['p0ymax']
    df['p1_pto_identif_x'] = (df['p1xmax']+df['p1xmin'])/2
    df['p1_pto_identif_y'] = df['p1ymax']
    df['p0_pto_real_x'] = 3.2
    df['p0_pto_real_y'] = 3.2
    df['p1_pto_real_x'] = 3.2
    df['p1_pto_real_y'] = 3.2
    df['p0_caja_posicion'] = 0
    df['p1_caja_posicion'] = 0
    M = homografia(puntosimagen)
    cajas = cajaspista()
    for x in range(len(df)):
        pts = np.array([df['p0_pto_identif_x'][x],df['p0_pto_identif_y'][x]], np.float32).reshape(-1, 1, 2)
        puntosenpista = cv2.perspectiveTransform(pts, M)
        df['p0_pto_real_x'][x] = puntosenpista[0][0][0]
        df['p0_pto_real_y'][x] = puntosenpista[0][0][1]
        pts1 = np.array([df['p1_pto_identif_x'][x],df['p1_pto_identif_y'][x]], np.float32).reshape(-1, 1, 2)
        puntosenpista1 = cv2.perspectiveTransform(pts1, M)
        df['p1_pto_real_x'][x] = puntosenpista1[0][0][0]
        df['p1_pto_real_y'][x] = puntosenpista1[0][0][1]
    
        for y in range(0,12):
            if cajas[y].contains(df['p0_pto_real_x'][x],df['p0_pto_real_y'][x]):
                df['p0_caja_posicion'][x] = y+1
        for p in range(12,24):
            if cajas[p].contains(df['p1_pto_real_x'][x],df['p1_pto_real_y'][x]):
                df['p1_caja_posicion'][x] = p+1
    df.to_csv('ZonaImagenesBBoxesTiemposPuntuacion'  + nombrevideo + '.csv',  index=False, encoding='utf-8')
    


# In[98]:


COURT_LINES = [
    [(0, 0), (609, 0)],
    [(0, 3), (609, 3)],
    [(0, 76), (609, 76)],
    [(0, 79), (609, 79)],
    [(0, 468), (609, 468)],
    [(0, 471), (609, 471)],
    [(0, 868), (609, 868)],
    [(0, 871), (609, 871)],
    [(0, 1260), (609, 1260)],
    [(0, 1263), (609, 1263)],
    [(0, 1339), (609, 1339)],
    [(0, 1336), (609, 1336)],
    [(0, 0), (0, 1339)],
    [(3, 0), (3, 1339)],
    [(46, 0), (46, 1339)],
    [(49, 0), (49, 1339)],
    [(303, 0), (303, 1339)],
    [(306, 0), (306, 1339)],
    [(560, 0), (560, 1339)],
    [(563, 0), (563, 1339)],
    [(606, 0), (606, 1339)],
    [(609, 0), (609, 1339)]
]


class BadmintonCourt(object):
    def __init__(self):
        pass

    @staticmethod
    def court_image():
        # Create a black image
        img = np.zeros((1340, 610, 1), np.uint8)

        # lineas horizontales de arriba a abajo
        for line in COURT_LINES:
            cv2.line(img, line[0], line[1], 255, 1)

        return img

    @staticmethod
    def court_lines():
        return COURT_LINES

    @staticmethod
    def court_corners():

        horizontal_lines = []
        vertical_lines = []

        # for each line we should estimate de line parameters y = ax + b; a and b are the parameters
        for line in COURT_LINES:

            if line[0][0] == line[1][0]:
                vertical_lines.append(line)
            else:
                horizontal_lines.append(line)

        corners = []
        for hoz_line in horizontal_lines:
            for vert_line in vertical_lines:
                corners.append((vert_line[0][0],hoz_line[0][1]))

        corners.sort(key = lambda point: point[0]+point[1])
        return corners

    @staticmethod
    def court_external_4_corners():
        #return [(0, 0),(0, 1339),(609, 1339), (609, 0)]
        return [(0, 1339), (609, 1339),(0, 0), (609, 0)]

    @staticmethod
    def court_medium_corners():
        aux = []
        for point in BadmintonCourt.court_corners():
            if point[1]> 669:
                aux.append(point)
        return aux

if __name__ == "__main__":
    b = BadmintonCourt()

    cv2.imwrite("pistareal.jpg", b.court_image(),
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])


# In[99]:


def draw_points(image, points, color_palette='tab20', palette_samples=16, confidence_threshold=0.5):
    """
    Draws `points` on `image`.
    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        color_palette: name of a matplotlib color palette
            Default: 'tab20'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5
    Returns:
        A new image with overlaid points
    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    circle_size = max(1, min(image.shape[:2]) // 160)  # ToDo Shape it taking into account the size of the detection
    # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))

    for i, pt in enumerate(points):
            image = cv2.circle(image, (int(pt[1]), int(pt[0])), circle_size, tuple(colors[i % len(colors)]), -1)

    return image





