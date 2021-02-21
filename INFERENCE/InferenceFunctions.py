#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2  # for capturing videos
import math  # for mathematical operations
import matplotlib.pyplot as plt  # for plotting the images

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from keras.preprocessing import image  # for preprocessing the images
import numpy as np  # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize  # for resizing images
import argparse
import os
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
import struct
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from skimage.transform import resize
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from skimage.color import gray2rgb
from sklearn.cluster import KMeans
import itertools
from sklearn import mixture
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
import struct
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from skimage.transform import resize
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import keras
from keras.preprocessing.image import ImageDataGenerator
import pickle
from skimage.feature import hog
import joblib
# In[10]:
from skimage import io, transform, exposure, color, util, filters
import random
np.random.seed(42)
import os
import tqdm
from matplotlib.transforms import Bbox
from sklearn import preprocessing
import matplotlib.image as mpimg
from os import remove
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import shutil
import os
path = os.getcwd()
# Random words for HOG features columns
wordlist = [ 'closet',
'clue',
'collar',
'comment',
'conference',
'devil',
'diet',
'fear',
'fuel',
'glove',
'jacket',
'lunch',
'monitor',
'mortgage',
'nurse',
'pace',
'panic',
'peak',
'plane',
'reward',
'row',
'sandwich',
'shock',
'spite',
'spray',
'surprise',
'till',
'transition',
'weekend',
'welcome',
'yard',
'alarm',
'bend',
'bicycle',
'bite',
'blind',
'bottle',
'cable',
'candle',
'clerk',
'cloud',
'concert',
'counter',
'flower',
'grandfather',
'harm',
'knee',
'lawyer',
'leather',
'load',
'mirror',
'neck',
'pension',
'plate',
'purple',
'ruin',
'ship',
'skirt',
'slice',
'snow',
'specialist',
'stroke',
'switch',
'trash',
'tune',
'zone',
'anger',
'award',
'bid',
'bitter',
'boot',
'bug',
'camp',
'candy',
'carpet',
'cat',
'champion',
'channel',
'clock',
'comfort',
'cow',
'crack',
'engineer',
'entrance',
'fault',
'grass',
'guy',
'hell',
'highlight',
'incident',
'island',
'joke',
'jury',
'leg',
'lip',
'mate',
'motor',
'nerve',
'passage',
'pen',
'pride',
'priest',
'prize',
'promise',
'resident',
'resort',
'ring',
'roof',
'rope',
'sail',
'scheme',
'script',
'sock',
'station',
'toe',
'tower',
'truck',
'witness',
'a',
'you',
'it',
'can',
'will',
'if',
'one',
'many',
'most',
'other',
'use',
'make',
'good',
'look',
'help',
'go',
'great',
'being',
'few',
'might',
'still',
'public',
'read',
'keep',
'start',
'give',
'human',
'local',
'general',
'she',
'specific',
'long',
'play',
'feel',
'high',
'tonight',
'put',
'common',
'set',
'change',
'simple',
'past',
'big',
'possible',
'particular',
'today',
'major',
'personal',
'current',
'national',
'cut',
'natural',
'physical',
'show',
'try',
'check',
'second',
'call',
'move',
'pay',
'let',
'increase',
'single',
'individual',
'turn',
'ask',
'buy',
'guard',
'hold',
'main',
'offer',
'potential',
'professional',
'international',
'travel',
'cook',
'alternative',
'following']

def ExtractDataSetFromVideoINFERENCE(nombrevideo):
    count = 0
    videoFile = nombrevideo + '.mp4'
    cap = cv2.VideoCapture(videoFile)  # capturing the video from the given path
    frameRate = cap.get(5)  # frame rate
    success = True
    pp = []
    fileID = []
    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()

        if (ret != True):
            break
        '''cv2.imwrite('kang'+str(i)+'.jpg',frame)'''
        if (frameId % math.floor(5) == 0):
            filename = nombrevideo + "frame%d.jpg" % count;
            count += 1
            pp.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            fileID.append(filename)
            cv2.imwrite(path + '/imagenesPARTIDO/CLASE0/' + filename, frame)
    cap.release()
    print("Done!")

    pp = np.array(pp)
    pp = pp / 1000
    df = pd.DataFrame(list(zip(fileID, pp)), columns=['Filename', 'segundo'])
    filename = 'FramesYTiempos' + nombrevideo + '.csv'
    df.to_csv(filename, index=False, encoding='utf-8')


# In[4]:




def identifygameimagesINFERENCE(nombrevideo):
    model = load_model('modeloVGG1623augweightprueba2.h5')
    model2 = joblib.load('my_model_knn.pkl')
    nombreframescsv = 'FramesYTiempos' + nombrevideo + '.csv'
    data = pd.read_csv(nombreframescsv)
    datasetimg = pd.DataFrame(data)
    datasetimg['juego'] = np.zeros(len(datasetimg))

    ruta_coleccion = 'imagenesPARTIDO/CLASE0/*.jpg'

    prediction = []
    predictionknear = []
    for x in range(len(datasetimg)):

        img = cv2.imread('imagenesPARTIDO/CLASE0/' + datasetimg['Filename'][x])
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img) / 255

        fd, _ = hog(img, orientations=8, pixels_per_cell=(16, 16), block_norm='L2', cells_per_block=(1, 1),visualize=True, multichannel=True)
        predictionknear.append(model2.predict(fd.reshape(1, -1)))
        prediction.append(model.predict(img[np.newaxis, :, :, :]))
        #datasetimg['juego'][x] = prediction

    predictionknear2 = np.array(predictionknear)
    predictionknear2 = np.concatenate(predictionknear2)
    prediction2 = np.concatenate(np.concatenate(prediction))
    alpha = 0.45
    voting = prediction2 * (1 - alpha) + predictionknear2 * alpha
    pp = []
    for x in voting:
        if x >= 0.5:
            pp.append(1)
        else:
            pp.append(0)
    predicted_class_voting = np.array(pp)
    datasetimg['juego'] = predicted_class_voting
    filename2 = 'identificarjuegoono' + nombrevideo + '.csv'
    datasetimg.to_csv(filename2, index=False, encoding='utf-8')




def separategameimagesINFERENCE(nombrevideo):
    filename2 = 'identificarjuegoono' + nombrevideo + '.csv'
    data = pd.read_csv(filename2)
    datasetimg = pd.DataFrame(data)

    for x in range(len(datasetimg)):
        if datasetimg["juego"][x] == 0:
            filename = datasetimg["Filename"][x]
            shutil.move(path + '/imagenesPARTIDO/CLASE0/' + filename,
                        path + '/imagenesjuegoPARTIDO' + '/imagenesNOJUEGO')
        elif datasetimg["juego"][x] == 1:
            filename = datasetimg["Filename"][x]
            shutil.move(path + '/imagenesPARTIDO/CLASE0/' + filename, path + '/imagenesjuegoPARTIDO' + '/imagenesjuego')
    data2 = pd.read_csv(filename2)
    df = pd.DataFrame(data2)
    df2 = df[df.juego == 1].reset_index(drop=True)
    df2.to_csv('Soloimagenesjuego' + nombrevideo + '.csv', index=False, encoding='utf-8')



import cv2
import numpy as np
import glob


def videoimagesgame(nombrevideo):
    frameSize = (1280, 720)
    nombreidentificarjuego = 'Soloimagenesjuego' + nombrevideo + '.csv'
    data2 = pd.read_csv(nombreidentificarjuego)
    df = pd.DataFrame(data2)
    out = cv2.VideoWriter('juego' + nombrevideo + '.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 5, frameSize)

    for filename in df['Filename']:
        photofilename = 'imagenesjuegoPARTIDO/imagenesjuego/' + filename
        img = cv2.imread(photofilename)
        out.write(img)

    out.release()


# In[1]:







def rellenarBboxes(nombrevideo):
    nombrearchivoDS = 'DSImagenesBBoxesTiemposPuntuacion' + nombrevideo + '.csv'
    data3 = pd.read_csv(nombrearchivoDS)
    df = pd.DataFrame(data3)
    # Marcar las primeras imagenes de cada punto

    for x in df.index:
        if os.path.isfile(path + '/imagenesjugadoresPARTIDO/' + df.p1filename[x]):
            pass
        else:
            imarray = np.random.rand(100, 100, 3) * 255
            cv2.imwrite(path + '/imagenesjugadoresPARTIDO/' + df.p1filename[x], imarray)

    for x in df.index:
        if os.path.isfile(path + '/imagenesjugadoresPARTIDO/' + df.p0filename[x]):
            pass
        else:
            imarray = np.random.rand(100, 100, 3) * 255
            cv2.imwrite(path + '/imagenesjugadoresPARTIDO/' + df.p0filename[x], imarray)


# In[ ]:


def indetifyTopBottomPlayer(nombrepartido):
    nombrearchivocsv = 'DSImagenesBBoxesTiemposPuntuacion' + nombrepartido + '.csv'
    data = pd.read_csv(nombrearchivocsv)
    df = pd.DataFrame(data)
    juegovideo = 'juego' + nombrepartido + '.mp4'
    cap = cv2.VideoCapture(juegovideo)

    fgbg = cv2.createBackgroundSubtractorMOG2()
    df['p0filenamemask'] = 'p'
    df['p1filenamemask'] = 'p'
    count = 0
    while (1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)
        img = cv2.imread(path + '/imagenesjuegoPARTIDO/imagenesjuego/' + df['Filename'][count])
        mask = fgmask
        mask2 = np.where((mask < 200), 0, 1).astype('uint8')
        # mask2 = np.where((mask==0),0,1).astype('uint8')
        mask3 = gray2rgb(mask2)
        img2 = img * mask3
        # cropjug0 = img2[int(df.p0ymin[count]):int(df.p0ymax[count]), int(df.p0xmin[count]):int(df.p0xmax[count])]
        filenamejug0 = 'fgmask' + df['p0filename'][count]
        df['p0filenamemask'][count] = filenamejug0
        filenamejug1 = 'fgmask' + df['p1filename'][count]
        df['p1filenamemask'][count] = filenamejug1
        if pd.notnull(df.p0ymin[count]):
            cropjug0 = img2[int(df.p0ymin[count]):int(df.p0ymax[count]), int(df.p0xmin[count]):int(df.p0xmax[count])]
            cv2.imwrite(path + '/imagenesjugadoresmask/' + filenamejug0, cropjug0)
        if pd.notnull(df.p1ymin[count]):
            cropjug1 = img2[int(df.p1ymin[count]):int(df.p1ymax[count]), int(df.p1xmin[count]):int(df.p1xmax[count])]
            cv2.imwrite(path + '/imagenesjugadoresmask/' + filenamejug1, cropjug1)
        # filename2 =   ('jugador%d' % count) + filename ;count+=1
        # cv2.imwrite(filename2, crop_img)

        # filename2 = 'fgmask'  + ('jugador0frame%d' % count) + '.jpg' ; count+=1

        count += 1
        if count == len(df):
            break

    # cv2.imshow('frame',fgmask)
    # plt.show()

    cap.release()
    pp = df.index[df['inicio_punto'] == 1]
    pp1 = np.array(pp)
    pp1 = np.append(pp1, len(df) - 1)
    pp1.shape
    tt = []
    for x in range(1, len(pp1)):
        if abs(pp1[x - 1] - pp1[x]) >= 10:
            tt.append(random.sample(range(pp1[x - 1], pp1[x]), 10))
        if abs(pp1[x - 1] - pp1[x]) < 10:
            rango = abs(pp1[x - 1] - pp1[x])
            print(rango)
            tt.append(random.sample(range(pp1[x - 1], pp1[x]), rango))
    tt = np.array(tt)

    clt = KMeans(n_clusters=5)
    avghistot1 = []
    for x in tt:
        avghis1 = []
        for y in x:
            hist = []
            photofilename = df['p1filenamemask'][y]
            if os.path.isfile(path + '/imagenesjugadoresmask/' + photofilename):
                img = cv2.imread(path + '/imagenesjugadoresmask/' + photofilename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.reshape((img.shape[0] * img.shape[1], 3))

                clt.fit(img)
                prp = clt.cluster_centers_
                # prp = prp.mean(axis=1)
                hist.append(prp)
                s = np.array(hist)
                avghis1.append(s)
        avghis1 = np.array(avghis1)
        u = avghis1.mean(axis=0)
        avghistot1.append(u)
    clt = KMeans(n_clusters=5)
    avghistot0 = []
    for x in tt:
        avghis0 = []
        for y in x:
            hist = []
            photofilename = df['p0filenamemask'][y]
            if os.path.isfile(path + '/imagenesjugadoresmask/' + photofilename ):
                img = cv2.imread(path + '/imagenesjugadoresmask/' + photofilename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.reshape((img.shape[0] * img.shape[1], 3))
                clt.fit(img)
                prp = clt.cluster_centers_
                # prp = prp.mean(axis=1)
                hist.append(prp)
                s = np.array(hist)
                avghis0.append(s)
        avghis0 = np.array(avghis0)
        u = avghis0.mean(axis=0)
        avghistot0.append(u)
    avghistot = avghistot0 + avghistot1
    for i in range(len(avghistot)):
        avghistot[i] = avghistot[i].ravel()
    # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=2,
                                            covariance_type='full').fit(avghistot)
    labels2 = dpgmm.predict(avghistot)
    df['pb0estaenpb'] = 3
    count = 0
    for x in range(len(pp1) - 1):
        pt = range(pp1[x], pp1[x + 1])
        df['pb0estaenpb'][pt] = labels2[count]
        count += 1
    df['pb0estaenpb'][len(df) - 1] = df['pb0estaenpb'][len(df) - 2]
    df.to_csv('DSMaskImagenesBBoxesTiemposPuntuacion' + nombrepartido + '.csv', index=False, encoding='utf-8')

def homografia(puntosimagen):
  esquinas_pista_reales = [(0, 0),(0, 1339),(609, 0), (609, 1339)]
  #Hacer homografía
  src_pts = np.array(esquinas_pista_reales, np.float32)
  dst_pts = np.array(puntosimagen, np.float32)
  # TODO ordenar puntos en el mismo orden.
  M, mask = cv2.findHomography(src_pts, dst_pts)
  inv_M = np.linalg.pinv(M)
  return inv_M

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

puntosimagen = [(217.5, 684), (438.5, 374), (1074, 684), (841.0, 374)]
def includecourtzonesandcreatedataset(puntosimagen,nombrevideo):
    nombrearchivocsv = path  + '/DSMaskImagenesBBoxesTiemposPuntuacion' + nombrevideo + '.csv'
    data = pd.read_csv(nombrearchivocsv)
    df = pd.DataFrame(data)
    df['p0_pto_identif_x'] = (df['p0xmax'] + df['p0xmin']) / 2
    df['p0_pto_identif_y'] = df['p0ymax']
    df['p1_pto_identif_x'] = (df['p1xmax'] + df['p1xmin']) / 2
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
        pts = np.array([df['p0_pto_identif_x'][x], df['p0_pto_identif_y'][x]], np.float32).reshape(-1, 1, 2)
        puntosenpista = cv2.perspectiveTransform(pts, M)
        df['p0_pto_real_x'][x] = puntosenpista[0][0][0]
        df['p0_pto_real_y'][x] = puntosenpista[0][0][1]
        pts1 = np.array([df['p1_pto_identif_x'][x], df['p1_pto_identif_y'][x]], np.float32).reshape(-1, 1, 2)
        puntosenpista1 = cv2.perspectiveTransform(pts1, M)
        df['p1_pto_real_x'][x] = puntosenpista1[0][0][0]
        df['p1_pto_real_y'][x] = puntosenpista1[0][0][1]

        for y in range(0, 12):
            if cajas[y].contains(df['p0_pto_real_x'][x], df['p0_pto_real_y'][x]):
                df['p0_caja_posicion'][x] = y + 1
        for p in range(12, 24):
            if cajas[p].contains(df['p1_pto_real_x'][x], df['p1_pto_real_y'][x]):
                df['p1_caja_posicion'][x] = p + 1
    df.to_csv('ZonaMaskImagenesBBoxesTiemposPuntuacion' + nombrevideo + '.csv', index=False, encoding='utf-8')
    nombrearchivocsv = path +'/ZonaMaskImagenesBBoxesTiemposPuntuacion' + nombrevideo + '.csv'
    data = pd.read_csv(nombrearchivocsv)
    df11 = pd.DataFrame(data)
    nombrearchivocsv = path +'/humanposep0' + nombrevideo + '.csv'
    data = pd.read_csv(nombrearchivocsv, index_col=0)
    df_hp0 = pd.DataFrame(data)
    nombrearchivocsv = path +'/humanposep1' + nombrevideo + '.csv'
    data = pd.read_csv(nombrearchivocsv, index_col=0)
    df_hp1 = pd.DataFrame(data)
    datasetdef = pd.concat([df11, df_hp0, df_hp1], axis=1)
    datasetdef.to_csv(path +'/CompletoDataset' + nombrevideo + '.csv', index=False,
                      encoding='utf-8')

def includehogfeatures(nombrevideo):
    nombrearchivocsv = path + '/CompletoDataset' + nombrevideo + '.csv'
    data = pd.read_csv(nombrearchivocsv)
    df = pd.DataFrame(data)

    img = []
    for p in df.p0filename:
        if os.path.isfile(path + '/imagenesjugadoresPARTIDO/' + '' + p):
            img.append(mpimg.imread(path + '/imagenesjugadoresPARTIDO/' + '' + p))
        else:
            print(p)


    img1 = []
    for p in df.p1filename:
        if os.path.isfile(path + '/imagenesjugadoresPARTIDO/' + '' + p):
            img1.append(mpimg.imread(path + '/imagenesjugadoresPARTIDO/' + '' + p))
        else:
            print(p)
    winSize = (64, 64)
    blockSize = (64, 64)
    blockStride = (8, 8)
    cellSize = (64, 64)
    nbins = 9
    hoog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    ppc = 16

    hog_features = []
    for image in tqdm(img):
        image2 = cv2.resize(image, (224, 224))
        winStride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)
        hist = hoog.compute(image2)
        hog_features.append(hist)


    hog_features11 = []
    for image in tqdm(img1):
        image2 = cv2.resize(image, (224, 224))
        winStride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)
        hist = hoog.compute(image2)
        hog_features11.append(hist)

    hog_features = np.array(hog_features)
    hog_features11 = np.array(hog_features11)

    hog_features = hog_features.reshape((len(df), 81))
    hog_features11 = hog_features11.reshape((len(df), 81))
    dshg = pd.DataFrame(hog_features)
    dshg1 = pd.DataFrame(hog_features11)
    dshgdef = pd.concat([dshg, dshg1], axis=1)
    dshgdef.columns = wordlist[0:162]
    dshgdef.reset_index(drop=True)
    dshgdef.to_csv('hogfeatures' + nombrevideo + '.csv', index=False, encoding='utf-8')

def strokepredict(nombrevideo):
    nombrearchivocsv = path + '/CompletoDataset' + nombrevideo + '.csv'
    data = pd.read_csv(nombrearchivocsv)
    dcasif = pd.DataFrame(data)
    nombrearchivocsv = path + '/hogfeatures' + nombrevideo + '.csv'
    data = pd.read_csv(nombrearchivocsv)
    dfhog = pd.DataFrame(data).reset_index(drop=True)
    df = pd.concat([dcasif, dfhog], axis=1)
    df['centroid_x_p0'] = (df.p0_right_hip_x + df.p0_left_hip_x) / 2
    df['centroid_y_p0'] = (df.p0_right_hip_y + df.p0_left_hip_y) / 2
    df['centroid_x_p1'] = (df.p1_right_hip_x + df.p1_left_hip_x) / 2
    df['centroid_y_p1'] = (df.p1_right_hip_y + df.p1_left_hip_y) / 2
    pp = range(36, 87, 3)
    pp1 = range(87, len(dcasif.columns) - 3, 3)

    for x in pp:
        df[df.columns[x + 2]] = np.sqrt(
            (df.iloc[:, x] - df['centroid_x_p0']) ** 2 + (df.iloc[:, x + 1] - df['centroid_y_p0']) ** 2)


    for x in pp1:
        df[df.columns[x + 2]] = np.sqrt(
            (df.iloc[:, x] - df['centroid_x_p1']) ** 2 + (df.iloc[:, x + 1] - df['centroid_y_p1']) ** 2)
    df2 = df.drop(['Filename', 'juego', 'p2xmin', 'p2xmax', 'p2ymin', 'p2ymax', 'p2score', 'p0filename', 'p1filename',
                   'p2filename', 'pb0estaenpb', 'p0_pto_identif_x', 'p0_pto_identif_y',
                   'p1_pto_identif_x', 'p1_pto_identif_y','p0filenamemask', 'p1filenamemask',
                   ], axis=1)
    #df3 = pd.get_dummies(df2, columns=["p0_caja_posicion", "p1_caja_posicion"])
    df4 = df3.dropna(axis=0).reset_index(drop=True)
    #df6 = df4.drop(df4[(df4['p0_caja_posicion_0'] == 1) & (df4['inicio_punto'] == 0)].index).reset_index(drop=True)
    #df8 = df6.drop(df6[(df6['p1_caja_posicion_0'] == 1) & (df6['inicio_punto'] == 0)].index).reset_index(drop=True)
    columnasstandar = ['p0xmin', 'p0xmax', 'p0ymin', 'p0ymax', 'p1xmin',
                       'p1xmax', 'p1ymin', 'p1ymax', 'centroid_x_p0', 'centroid_y_p0', 'centroid_x_p1', 'centroid_y_p1']
    columnasstandar2 = df8.columns[12:118]
    columnasminmax = ['segundo', 'p0score', 'p1score', 'inicio_punto']

    scaler = preprocessing.StandardScaler()
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    def scaleColumns(dff, cols_to_scale):
        for col in cols_to_scale:
            dff[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(dff[col])), columns=[col])
        return dff

    def minmaxColumns(dff, cols_to_scale):
        for col in cols_to_scale:
            dff[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(dff[col])), columns=[col])
        return dff

    # scaled_df = scaleColumns(df5,df5.columns)
    scaled_df = scaleColumns(df8, columnasstandar)
    scaled_df = scaleColumns(scaled_df, columnasstandar2)
    scaled_df = minmaxColumns(scaled_df, columnasminmax)
    scaled_df


