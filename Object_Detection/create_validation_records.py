### This file creates protobuf records needed for validation set of object detection model per
### instructions of tensorflow object detection

import augmentImage
import tensorflow as tf
import cv2

import os
import io
from object_detection.utils import dataset_util
from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS



def get_class_map():
    dirs = sorted(os.listdir('./Data'))
    return dict(zip(dirs, range(1,len(dirs)+1)))


def create_tf_example(img_jpg, box, class_id, class_name):

    encoded_jpg = io.BytesIO()
    img_jpg.save(encoded_jpg, 'jpeg')
    encoded_jpg = encoded_jpg.getvalue()
#     encoded_jpg_io = io.BytesIO(img_jpg)
#     image = Image.open(encoded_jpg_io)
    image = img_jpg
    width, height = image.size
    
    

    filename = "".encode('utf-8') # Filename of the image. Empty if image is not from file
    encoded_image_data = encoded_jpg # Encoded image bytes
    image_format = b'jpeg' # b'jpeg' or b'png'

    xmins = [box[0][0]/width] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [box[1][0]/width] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [box[0][1]/height] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [box[1][1]/height] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [class_name.encode('utf-8')] # List of string class name of bounding box (1 per box)
    classes = [class_id] # List of integer class id of bounding box (1 per box)

#     tf_example = tf.train.Example(features=tf.train.Features(feature={
#       'image/height': dataset_util.int64_feature(height),
#       'image/width': dataset_util.int64_feature(width),
#       'image/filename': dataset_util.bytes_feature(filename),
#       'image/source_id': dataset_util.bytes_feature(filename),
#       'image/encoded': dataset_util.bytes_feature(encoded_image_data),
#       'image/format': dataset_util.bytes_feature(image_format),
#       'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
#       'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
#       'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
#       'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
#       'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
#       'image/object/class/label': dataset_util.int64_list_feature(classes),
#     }))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
    

def main(_):
    class_map = get_class_map()
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    index = 1
    for parent, __, files in os.walk('./Data'):
        if parent == './Data':
            continue
   
        class_name = parent.split('/')[-1]
        print('*' * 10 , class_name)
        index += 1
        images, boxes =  augmentImage.augment(parent, './background', num=20)
        for i in range(len(images)):
            tf_example = create_tf_example(images[i], boxes[i] , class_map[class_name], class_name)
            writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.app.run()