#!/usr/bin/env python
# coding: utf-8

# In[10]:


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


# In[11]:


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

get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


path = os.getcwd()
path


# In[13]:


cv2.__version__


# # Crear video con solo imagenes de Juego

# In[1]:


import cv2
import numpy as np
import glob
import shutil


# In[17]:


# Load Yolo
net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# In[42]:


class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1
 
	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)
 
		return self.label
 
	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]
 
		return self.score
def _sigmoid(x):
	return 1. / (1. + np.exp(-x))
def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	netout[..., :2]  = _sigmoid(netout[..., :2])
	netout[..., 4:]  = _sigmoid(netout[..., 4:])
	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	netout[..., 5:] *= netout[..., 5:] > obj_thresh
 
	for i in range(grid_h*grid_w):
		row = i / grid_w
		col = i % grid_w
		for b in range(nb_box):
			# 4th element is objectness score
			objectness = netout[int(row)][int(col)][b][4]
			if(objectness.all() <= obj_thresh): continue
			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w # center position, unit: image width
			y = (row + y) / grid_h # center position, unit: image height
			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
			classes = netout[int(row)][col][b][5:]
			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
			boxes.append(box)
	return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
	new_w, new_h = net_w, net_h
	for i in range(len(boxes)):
		x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
		y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
		boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
		boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
		boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
		boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
	return boxes
def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b
	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			 return 0
		else:
			return min(x2,x4) - x3
 
def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	return float(intersect) / union
 
def do_nms(boxes, nms_thresh):
	if len(boxes) > 0:
		nb_class = len(boxes[0].classes)
	else:
		return
	for c in range(nb_class):
		sorted_indices = np.argsort([-box.classes[c] for box in boxes])
		for i in range(len(sorted_indices)):
			index_i = sorted_indices[i]
			if boxes[index_i].classes[c] == 0: continue
			for j in range(i+1, len(sorted_indices)):
				index_j = sorted_indices[j]
				if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
					boxes[index_j].classes[c] = 0
	return boxes    

def draw_boxes(filename, v_boxes, v_labels, v_scores):
	# load the image
	photofilename2 = path + '/imagenesjuegoPARTIDO/imagenesjuego/' + filename
	data = pyplot.imread(photofilename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = []
	ax = pyplot.gca()
	# plot each box
	box = []
	for i in range(len(v_boxes)):
		box = v_boxes[i]
		# get coordinates
		y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
		#print(x1, y1, x2, y2)
		# calculate width and height of the box
		width, height = x2 - x1, y2 - y1
		# create the shape
		rect = Rectangle((x1, y1), width, height, fill=False, color='white')
		# draw the box
		ax.add_patch(rect)
		# draw text and score in top left corner
		label = "%s (%.3f)" % (v_labels[i], v_scores[i])
		pyplot.text(x1, y1, label, color='white')
	# show the plot
	pyplot.savefig('Bboxes' + filename)
	pyplot.show()

def get_boxes(boxes, labels, thresh):
	v_boxes, v_labels, v_scores = list(), list(), list()
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			if box.classes[i] > thresh:
				v_boxes.append(box)
				v_labels.append(labels[i])
				v_scores.append(box.classes[i]*100)
				# don't break, many labels may trigger for one box
	return v_boxes, v_labels, v_scores


# # Función para sacar Bounding Boxex

# In[16]:


def extractmodifyimage(frame):
    img = img_to_array(frame)
    img = img.astype('float32')
    img /= 255.0
    # border widths; I set them all to 150
    top = 8
    bottom = 8
    left = 0
    right = 0
    crop = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    crop[0:210,:] = 0
    nrows, ncols,_ = crop.shape
    row, col = np.ogrid[:nrows, :ncols]
    outer_disk_mask4 =  (-row - 10.80934*col + 0.02371029*col**2 - 0.00002385233*col**3 + 9.317318e-9*col**4 - 1.19122e-24*col**5 > -2043.16)
    crop[outer_disk_mask4] = 0
    return crop


# In[17]:


anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


# In[20]:





def prediccionYOLO(filename, imagenmodificada,draw = True, threshold = 0.5): # Entra imagen sin modificar y la imagen modificada y sale un dataset con la BBoxes de los dos jugadores
    crop3 = cv2.resize(imagenmodificada, (416,416))
    blob = cv2.dnn.blobFromImage(crop3, 1, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    input_w, input_h = 416, 416
    image_w, image_h = 1280,720
    class_threshold = threshold
    class_ids = []
    confidences = []
    boxes = []
    tt = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = (detection[0])
                center_y = (detection[1])
                w = (detection[2])
                h = (detection[3])
                # Rectangle coordinates
                x = (center_x - w / 2)
                y = (center_y - h / 2)
                tt.append(scores)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    boxes1 = []
    for x in range(len(boxes)):
        box1 = BoundBox(boxes[x][0], boxes[x][1], boxes[x][0]+boxes[x][2], boxes[x][1]+boxes[x][3], objness = None, classes = tt[x] )
        boxes1.append(box1)
    boxes1 = correct_yolo_boxes(boxes1, image_h, image_w, input_h, input_w)
    boxes1 = do_nms(boxes1, 0.5)
    v_boxes, v_labels, v_scores = get_boxes(boxes1, labels, class_threshold)
    if draw == True:
        draw_boxes(filename, v_boxes, v_labels, v_scores)
    p = []
    for x in range(len(v_boxes)):
        l = [v_boxes[x].xmin,v_boxes[x].xmax,v_boxes[x].ymin,v_boxes[x].ymax,v_labels[x],v_scores[x]]
        p.append(l)
    resultados = pd.DataFrame(p, columns = ['xmin','xmax', 'ymin', 'ymax', 'clase', 'score'])
    resultados = resultados[resultados['clase'] == 'person'].reset_index(drop=True)
    resultados = resultados.sort_values('ymax',ascending=False).reset_index(drop=True)
    
    # draw the box
    return resultados


# In[4]:


def pintarBBoxes(resultados,filename):
    photofilename = path + '/imagenesjuegoPARTIDO/imagenesjuego/' + filename
    img = cv2.imread(photofilename,cv2.IMREAD_COLOR)
    count1 = 0
    for s in resultados.index:
        if s <= 3:
            crop_img = img[resultados.ymin[s]:resultados.ymax[s], resultados.xmin[s]:resultados.xmax[s]]
            filename2 =  path +  '/imagenesjugadoresPARTIDO/'  + ('jugador%d' % count1) + filename ;count1 +=1
            cv2.imwrite(filename2, crop_img)


# In[36]:


import imutils
from imutils.video import FPS


# In[1]:


import os
path = os.getcwd()
def yolofromvideo(nombrevideo,pintarBboxesjug = False,draw = False, threshold=0.5):
    nombreidentificarjuego = 'Soloimagenesjuego' + nombrevideo+ '.csv'
    data2 = pd.read_csv(nombreidentificarjuego) 
    df2 = pd.DataFrame(data2)
    df2['resultados'] = 3
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # initialize the width and height of the frames in the video file
    W = None
    H = None

    i=0
    pbar = tqdm(total=len(df2))

    
    print("[INFO] accessing video stream...")
    videoFile = 'juego' + nombrevideo + '.mp4'
    vs = cv2.VideoCapture(videoFile) 
    writer = None
    fps = FPS().start()
    # loop over frames from the video file stream
    ipp = 0
    while (vs.isOpened()):
    # read the next frame from the file
        
        (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
        if not grabbed:
            break
    # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
        crop3 = extractmodifyimage(frame)
        blob = cv2.dnn.blobFromImage(crop3, 1, (416, 416),
                swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(ln)
        input_w, input_h = 416, 416
        image_w, image_h = 1280,720
        class_threshold = threshold
        class_ids = []
        confidences = []
        boxes = []
        tt = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = (detection[0])
                    center_y = (detection[1])
                    w = (detection[2])
                    h = (detection[3])
                    # Rectangle coordinates
                    x = (center_x - w / 2)
                    y = (center_y - h / 2)
                    tt.append(scores)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        boxes1 = []
        for x in range(len(boxes)):
            box1 = BoundBox(boxes[x][0], boxes[x][1], boxes[x][0]+boxes[x][2], boxes[x][1]+boxes[x][3], objness = None, classes = tt[x] )
            boxes1.append(box1)
        boxes1 = correct_yolo_boxes(boxes1, image_h, image_w, input_h, input_w)
        boxes1 = do_nms(boxes1, 0.5)
        if boxes1:
            v_boxes, v_labels, v_scores = get_boxes(boxes1, labels, class_threshold)
            p = []
            for x in range(len(v_boxes)):
                l = [v_boxes[x].xmin,v_boxes[x].xmax,v_boxes[x].ymin,v_boxes[x].ymax,v_labels[x],v_scores[x]]
                p.append(l)
            resultados = pd.DataFrame(p, columns = ['xmin','xmax', 'ymin', 'ymax', 'clase', 'score'])
            resultados = resultados[resultados['clase'] == 'person'].reset_index(drop=True)
            resultados = resultados.sort_values('ymax',ascending=False).reset_index(drop=True)
            df2['resultados'][i] = np.array(resultados)
            if pintarBboxesjug == True:
                pintarBBoxes(resultados, df2['Filename'][ipp])
            if draw == True:
                draw_boxes(filename, v_boxes, v_labels, v_scores)
            ipp +=1
        else:
            df2['resultados'][ipp] = np.array(1)
            ipp +=1
        
        
        fps.update()
        pbar.update(1)
        i +=1
        
    pbar.close()    
    # stop the timer and display FPS information
    vs.release()
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    df2.to_csv(path +'/'+ 'BboxesSoloimagenesjuego' + nombrevideo + '.csv', index=False, encoding='utf-8')


# In[ ]:


def generatebigdataset(nombrevideo,weightpista=861, heightpista=471):
    weightpista = weightpista
    heightpista = heightpista
    nombrearchivocoordenadascsv = 'BboxesSoloimagenesjuego'+ nombrevideo + '.csv'
    data = pd.read_csv(nombrearchivocoordenadascsv) 
    df = pd.DataFrame(data)

    pp = df['resultados']

    for x in range(len(pp)):
        pp[x] = pp[x].replace('[', '').replace(']','').replace('\n','').replace("'",'')

    df_coor = pp.apply(lambda x: pd.Series(x.split(' ')))

    for i in range(0, len(df_coor.columns)):
        df_coor.iloc[:,i] = pd.to_numeric(df_coor.iloc[:,i], errors='ignore')
        # errors='ignore' lets strings remain as 'non-null objects'

    df_coor = df_coor.drop(range(18,len(df_coor.columns)), axis = 1)

    df_coor.columns = ['p0xmin','p0xmax', 'p0ymin', 'p0ymax', 'p0clase', 'p0score','p1xmin','p1xmax', 'p1ymin', 'p1ymax', 'p1clase', 'p1score','p2xmin','p2xmax', 'p2ymin', 'p2ymax', 'p2clase', 'p2score']

    df_coor = df_coor.drop(['p0clase','p1clase','p2clase'], axis = 1)

    dfgrande= pd.concat([df,df_coor],axis=1)

    dfgrande = dfgrande.drop(['resultados'], axis = 1)

    dfgrande['p0filename'] = 'p'
    dfgrande['p1filename'] = 'p'
    dfgrande['p2filename'] = 'p'

    for x in dfgrande.index:
        dfgrande['p0filename'][x] = 'jugador0' + dfgrande['Filename'][x]
        dfgrande['p1filename'][x] = 'jugador1' + dfgrande['Filename'][x]
        dfgrande['p2filename'][x] = 'jugador2' + dfgrande['Filename'][x]

    # Bucle para evitar que haya identificado una Bboxes muy grande como persona
    # y desvirtue la correspondencia entre el jugador y su Bboxes 

    

    for x in dfgrande.index:
        if (dfgrande.p0xmax[x] - dfgrande.p0xmin[x])*(dfgrande.p0ymax[x] - dfgrande.p0ymin[x]) >= weightpista*heightpista:
            dfgrande.p0xmin[x] = dfgrande.p1xmin[x] 
            dfgrande.p0xmax[x] = dfgrande.p1xmax[x] 
            dfgrande.p0ymin[x] = dfgrande.p1ymin[x] 
            dfgrande.p0ymax[x] = dfgrande.p1ymax[x]
            dfgrande.p1xmin[x] = dfgrande.p2xmin[x] 
            dfgrande.p1xmax[x] = dfgrande.p2xmax[x] 
            dfgrande.p1ymin[x] = dfgrande.p2ymin[x] 
            dfgrande.p1ymax[x] = dfgrande.p2ymax[x] 
            dfgrande.p2xmin[x] = np.nan
            dfgrande.p2xmax[x] = np.nan
            dfgrande.p2ymin[x] = np.nan
            dfgrande.p2ymax[x] = np.nan
            #dfgrande2.p0filename[x] = dfgrande2.p1filename[x]
            #dfgrande2.p1filename[x] = dfgrande2.p2filename[x]
            if os.path.isfile(path + '/imagenesjugadoresPARTIDO/' +dfgrande.p0filename[x]): 
                shutil.move(path + '/imagenesjugadoresPARTIDO/' +dfgrande.p0filename[x], path + '/' + dfgrande.p0filename[x])
                
            old_file_name1 = path + '/imagenesjugadoresPARTIDO/' + dfgrande.p1filename[x]
            new_file_name1 = path + '/imagenesjugadoresPARTIDO/' + dfgrande.p0filename[x]
            old_file_name2 = path + '/imagenesjugadoresPARTIDO/' + dfgrande.p2filename[x]
            new_file_name2 = path + '/imagenesjugadoresPARTIDO/' + dfgrande.p1filename[x]
            if os.path.isfile(old_file_name1 ):
                os.rename(old_file_name1, new_file_name1)
            if os.path.isfile(old_file_name2 ):
                os.rename(old_file_name2, new_file_name2)
    for x in dfgrande.index:
        if (dfgrande.p1xmax[x] - dfgrande.p1xmin[x])*(dfgrande.p1ymax[x] - dfgrande.p1ymin[x]) >= weightpista*heightpista:
            dfgrande.p1xmin[x] = dfgrande.p2xmin[x] 
            dfgrande.p1xmax[x] = dfgrande.p2xmax[x] 
            dfgrande.p1ymin[x] = dfgrande.p2ymin[x] 
            dfgrande.p1ymax[x] = dfgrande.p2ymax[x] 
            dfgrande.p2xmin[x] = np.nan
            dfgrande.p2xmax[x] = np.nan
            dfgrande.p2ymin[x] = np.nan
            dfgrande.p2ymax[x] = np.nan
            if os.path.isfile(path + '/imagenesjugadoresPARTIDO/' +dfgrande.p1filename[x]): 
                shutil.move(path + '/imagenesjugadoresPARTIDO/' +dfgrande.p1filename[x],path +'/'+dfgrande.p1filename[x])
            old_file_name3 = path + '/imagenesjugadoresPARTIDO/' + dfgrande.p2filename[x]
            new_file_name3 = path + '/imagenesjugadoresPARTIDO/' + dfgrande.p1filename[x]
            if os.path.isfile(old_file_name2):
                os.rename(old_file_name3, new_file_name3)

        
    #Marcar las primeras imagenes de cada punto
    dfgrande['inicio_punto'] = 0
    dfgrande['inicio_punto'][0] = 1
    for x in range(1,len(dfgrande)-2):
        if (dfgrande['segundo'][x]-dfgrande['segundo'][x-1])> 2.0 and (dfgrande['segundo'][x+2]-dfgrande['segundo'][x]) <= 0.41 :
            dfgrande['inicio_punto'][x] = 1
    dfgrande.to_csv(path + '/'+'DSImagenesBBoxesTiemposPuntuacion' + nombrevideo + '.csv', index=False, encoding='utf-8')






