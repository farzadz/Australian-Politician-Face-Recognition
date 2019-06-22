import argparse
from keras.utils.generic_utils import get_custom_objects
from .custom_metrics import f1, precision, recall
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
from keras.models import load_model
import json
import os
import cv2


MODEL_FILE = 'Farzad-2Dense-255class-VGGFACE-improvement-13-0.92.hdf5'
CLASS_INDEX_FILE = 'classes.json'


current_dir = os.path.dirname(os.path.realpath(__file__))


model_path = os.path.join(current_dir, MODEL_FILE)
get_custom_objects().update({"f1": f1, 'precision': precision, 'recall': recall})
model = load_model(model_path)
model._make_predict_function()

classes_path = os.path.join(current_dir, CLASS_INDEX_FILE)
with open(classes_path) as cl:
    class_labels = json.load(cl)
    

def predict_label(image):
    "jpeg file path and json file containing index -> name as key value pairs"
    
    
    
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    image = img_to_array(image) / 255
    image = np.expand_dims(image, axis=0)
    prediction_tensor = model.predict_on_batch(image)
    most_likely = np.argmax(prediction_tensor)
    return class_labels[str(most_likely)]


# def predict_label_from_directory(path):
#     "jpeg file path and json file containing index -> name as key value pairs"
#     files = [os.path.join(path,file) for file in os.listdir(path)]

#     images = []
#     for image_file in files:
#         image = load_img(image_file, target_size=(224, 224))
#         image = img_to_array(image) / 255
#         images.append(image)
#     images = np.array(images)
# #     print('images np array dims: ', images.shape)

#     prediction_tensor = model.predict_on_batch(images)
#     most_likely = np.argmax(prediction_tensor,axis=1)
# #     print(most_likely)
#     return [(files[i] ,class_labels[str(most_likely[i])]) for i in range(len(files))]



# if __name__ == '__main__':
    
    
#     ap = argparse.ArgumentParser(description='Returns the name of an Australian politican given his/her image')
#     ap.add_argument("-i", "--imagedir", required=True,
#             help="path to images directory")

#     args = vars(ap.parse_args())
#     print(args)
#     print(predict_label_from_directory(args['imagedir']))