
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np

import FaceToolKit as ftk
import DetectionToolKit as dtk




image_size = 160
v = ftk.Verification()
# Pre-load model for Verification
v.load_model("./models/20180204-160909/")
v.initial_input_output_tensors()


d = dtk.Detection()




from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2


# In[5]:


def img_to_encoding(img):
    aligned = d.align(img, False)[0]
    return v.img_to_encoding(aligned, image_size)








import os
import logging
import matplotlib.pyplot as plt

logging.basicConfig(filename='pruning.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def prune_faces_from_dir(directory):
        
    if not os.path.exists('./cleaned'):
        os.makedirs('./cleaned')
        
    # dangerous!
    extracted_dir = os.path.join('./cleaned', directory.split('/')[-1])

    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir)
    for parent ,__, files in os.walk(directory):
        num_faces = 0
        images = []
        image_names = []
        encodings = []
        for file in files:
            try:
                 image = plt.imread(parent + '/' + file)
                 encodings.append(img_to_encoding(image))
                 images.append(image)
                 image_names.append(file)
            except Exception as e:
                  print(e)
                  logging.info('Image {} is corrupted'.format(file))
            
  
        clt = DBSCAN(eps=0.8, metric="euclidean", n_jobs=3)
        clt.fit(encodings)
        relevant_images = [images[i] for i in range(len(images)) if clt.labels_[i] == 0]
        for i, face in enumerate(relevant_images):

            plt.imsave(os.path.join(extracted_dir, str(i) + '.jpg'), face)
