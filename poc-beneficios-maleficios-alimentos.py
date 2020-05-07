
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2

#Captura Webcam Princiapl
cap = cv2.VideoCapture(0)

# Necessário somente se o notebook - código estiver na pasta object_detection.
sys.path.append("..")


# Importando módulos do Object Detection
from utils import label_map_util
from utils import visualization_utils as vis_util

#Preparando o Modelo

#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17' 
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Caminho para frozen detection graph. usado pelo Object Detection
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# Lista de variáveis usada para escrever os nomes corretos dos labels (ao prever os objetos). 
PATH_TO_LABELS = 'data/mscoco_modificado_label_map.pbtxt'
NUM_CLASSES = 90

#Download do Modelo

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

# Carregando o modelo em memória

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Carregando Label Map - Índices para os nomes das categorias, quando a rede neural preve id 5 que corresponde a airplane

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


#def load_image_into_numpy_array(image):
#  (im_width, im_height) = image.size
#  return np.array(image.getdata()).reshape(
#      (im_height, im_width, 3)).astype(np.uint8)

# Execução da detecção gráfica 
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      
      
      
          # Resultados da Predição
      if classes[0][0] == 52:
                    cv2.putText(image_np,'IDENTIFICAMOS A FRUTA BANANA:  ',(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                    cv2.putText(image_np, 'O principal beneficio da Banana!',(20,90),cv2.FONT_HERSHEY_SIMPLEX,0.5,(147,219,112),2)
                    cv2.putText(image_np, 'Rica em potássio, perfeita para baixar a pressão arterial.',(20,120),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                    
                    
                
      else:
                    cv2.putText(image_np,'Nao encontrado ',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
    
                
    

      cv2.imshow('FRUTAS E SEUS BENEFICIOS E MALEFICIOS AO ORGANISMO', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break