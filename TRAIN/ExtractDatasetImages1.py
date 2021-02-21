#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Base libraries
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

# Visual setup
plt.style.use('fivethirtyeight')   # alternatives below
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
custom_colors = ['#74a09e','#86c1b2','#98e2c6','#f3c969','#f2a553', '#d96548', '#c14953']
sns.set_palette(custom_colors)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# Pandas options
pd.set_option('max_colwidth', 40)
pd.options.display.max_columns = None  # Possible to limit
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# Seed value for numpy.random
np.random.seed(42)


# In[4]:


import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
import shutil
import os
from os import remove
path = os.getcwd()

# In[5]:



def ExtractDataSetFromVideo(namevideo, namefilecsv):
    count = 0
    videoFile = namevideo + '.mp4'
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    success = True
    pp = []
    fileID = []
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
    
        if (ret != True):
            break
        '''cv2.imwrite('kang'+str(i)+'.jpg',frame)'''
        if (frameId % math.floor(5) == 0):
            filename = namevideo + "frame%d.jpg" % count;count+=1
            pp.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            fileID.append(filename)
            cv2.imwrite(path + '/imagenesangle/'+ filename, frame)
    cap.release()
    print ("Done!")
    
    
    pp = np.array(pp)
    pp = pp/1000
    df = pd.DataFrame(list(zip(fileID, pp)), columns =['Filename', 'segundo']) 
    filename = 'FramesYTiempos' + namevideo   + '.csv'
    df.to_csv(filename, index=False, encoding='utf-8')
    
    '''Introducir el segundo dataset'''
    
    data = pd.read_csv(namefilecsv, header=None)
    df2 = pd.DataFrame(data)
    location_df = df2[8].apply(lambda x: pd.Series(x.split(',')))
    location_df.columns = ['p_saca','p_win','score1','score2','angle']
    dfdef= pd.concat([df2,location_df],axis=1)
    dfdef = dfdef.drop([0,1,8], axis = 1)
    dfdef.columns = ['t_0','t_0seg','t_final', 't_finalseg', 't_intervalo', 't_intervalo_seg','p_saca','p_win','score_1','score_2','angle']
    filename2 = 'datospartido' + nombrevideo  + '.csv'
    dfdef.to_csv(filename2, index=False, encoding='utf-8')


# In[6]:



def identifygameorangle(nombrevideo):
    nombreframescsv = 'FramesYTiempos' + nombrevideo   + '.csv'
    nombredatoscsv = 'datospartido' + nombrevideo  + '.csv'
    data = pd.read_csv(nombreframescsv) 
    datasetimg = pd.DataFrame(data)
    datasetimg['juego'] = np.zeros(len(datasetimg))
    data2 = pd.read_csv(nombredatoscsv) 
    df = pd.DataFrame(data2)
    df['angle'] = df['angle'].astype('str')
    df['angle'] = df['angle'].astype("string")
    for x in datasetimg.index:
        for y in df.index: 
            if  ('angle' in df['angle'][y]) and (datasetimg["segundo"][x] >= df['t_0seg'][y]) and (datasetimg["segundo"][x] <= df['t_finalseg'][y]):
                datasetimg['juego'][x] = 2
            elif (datasetimg["segundo"][x] >= df['t_0seg'][y]) and (datasetimg["segundo"][x] <= df['t_finalseg'][y]):
                datasetimg['juego'][x] = 1
    
    '''for l in datasetimg.index:
            if datasetimg['juego'][l] == 2:
                frametoelim = datasetimg['Filename'][l]
                remove(path + '/'+ frametoelim)
                datasetimg.drop(l,inplace=True)
    datasetimg = datasetimg.reset_index(drop=True)'''
    
    
    filename2 = 'identificarjuegoono' + nombrevideo + '.csv'
    datasetimg.to_csv(filename2, index=False, encoding='utf-8')
    


# In[7]:


def identifygameimages(nombrevideo):

    filename2 = 'identificarjuegoono' + nombrevideo  + '.csv'
    data = pd.read_csv(filename2) 
    datasetimg = pd.DataFrame(data)
    
    for x in range(len(datasetimg)):
        if datasetimg["juego"][x] == 0:
                filename = datasetimg["Filename"][x]
                shutil.move(path + '/imagenesangle/' + filename, path + '/imagenesjuegoono' + '/imagenesNOJUEGO')
        elif datasetimg["juego"][x] == 1:
                filename = datasetimg["Filename"][x]
                shutil.move(path + '/imagenesangle/' + filename, path + '/imagenesjuegoono' + '/imagenesjuego')






