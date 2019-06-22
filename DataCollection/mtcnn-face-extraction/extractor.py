import tensorflow as tf
import numpy as np
import cv2
from detection.mtcnn import detect_face
import matplotlib.pyplot as plt


default_color = (0, 255, 0) #BGR
default_thickness = 2





def calculate_degree(pt1,pt2):
    """Calculates the the angle that corresponds to the slope of the line connecing pt1 to pt2.
    
    Args:
        pt1 ((x,y) or [x,y]): Coordinate of the first point.
        pt2 ((x,y) or [x,y]): Coordinate of the second point.
        
    Returns:
        float: angle
    """
    
    pt1,pt2 = np.array(pt1), np.array(pt2)
    pt1, pt2 = pt1[:], pt2[:]
    dx = (pt1[0] - pt2[0])
    dy = (pt1[1] - pt2[1])
    angle = np.degrees(np.arctan2(dy, dx)) - 180
    return angle


def display_point(img, point):
    """Displays a point on an image without changing the original image.
    
    Args:
        img (nd-array): Image as a numpy array.
        point ((x,y) or [x,y]): Coordinate of the point.
        
    Returns:
        nd-array: Mutated image with the point on it.
    """
    img = img.copy()
    point = (point[0], point[1])
    cv2.circle(img, center=point, radius=1, color=default_color, thickness=default_thickness)
    plt.imshow(img)
    plt.show()
    return img
def display_box(img, box):
    """Displays a rectangle on an image without changing the original image.
    
    Args:
        img (nd-array): Image as a numpy array.
        box (nd-array): Array of shape 2*2 [[x1,y1], [x2,y2]], where x1,y1 represent the point 
                        to the lower left corner of the rectangle, and [x2,y2] represent upper-right one.
        
    Returns:
        nd-array: Mutated image with the box shown.
    """
    box = box.astype('int32')
    #box should have (x,y) coords stacked vertically
    img = img.copy()
    cv2.rectangle(img, (box[0,0], box[0,1]), (box[1,0], box[1,1]), color=default_color, thickness=default_thickness)
    plt.imshow(img)
    plt.show()
    return img

def scale_box(box, scale):
    """Scales a rectangle by a ratio.
    
    Args:
        box (nd-array): Array of shape 2*2 [[x1,y1], [x2,y2]], where x1,y1 represent the point 
                        to the lower left corner of the rectangle, and [x2,y2] represent upper-right one.
        scale (float): Ratio to which the box is scaled.
        
    Returns:
        (nd-array): 2*2 Array with the same properties as box argument.
        """
    #upper left and lower-right as (x,y) stacked vertically
    center = np.mean(box,axis=0).astype('int32')
    h = abs(box[0,1] - box[1,1]) * scale
    w = abs(box[0,0] - box[1,0]) * scale
    return np.array([[center[0]-w//2, center[1] - h//2], [center[0] + w//2, center[1] + h//2]]).astype('int32')
    
def safe_crop(img, box):
    """Crops an image with handling out of bound coordinates based on a rectangular area.
    
    Args:
        img (nd-array): Image as a numpy array. 
        box (nd-array): Array of shape 2*2 [[x1,y1], [x2,y2]], where x1,y1 represent the point 
                        to the lower left corner of the rectangle, and [x2,y2] represent upper-right one.
    Returns:
        (nd-array): Cropped image.
    """
    h, w = img.shape[0], img.shape[1]
    
    # box is invalid:
    if box[0,0] > box[1,0] or box[0,1] > box[1,1]:
        return None
    # coords are beyond image boundaries:
    box[0,0] = 0 if box[0,0] < 0 else box[0,0]
    box[0,0] = w if box[0,0] > w else box[0,0]
    box[1,0] = 0 if box[1,0] < 0 else box[1,0]
    box[1,0] = w if box[1,0] > w else box[1,0]
    box[0,1] = 0 if box[0,1] < 0 else box[0,1]
    box[0,1] = h if box[0,1] > h else box[0,1]
    box[1,1] = 0 if box[1,1] < 0 else box[1,1]
    box[1,1] = w if box[1,1] > h else box[1,1]
    return img[box[0,1]:box[1,1], box[0,0]:box[1,0]]

def apply_affine(points, transformation):
    """Applies an affine transformation (like the output of cv2.getRotationMatrix2D) on a vector or matrix.
    
    Args:
        points (nd-array): Array of form [[x1,y1], [x2,y2], ...] stacked vertically.
        transformation (nd-array): A 2*3 affine transformation matrix.
        
    Returns:
        (nd-array): Points under the influence of the transformation, with the same format as input.
    """
    return ((np.dot(transformation[:,:2], points.reshape(-1, 2).T) + transformation[:,2].reshape(2,1)).T).astype('int32')

def scale_points(points,img_source, img_dest):
    """Scales points of source image to correspond to their position in destination.
    
    Args:
        points (nd-array): Array of form [[x1,y1], [x2,y2], ...] stacked vertically.
        img_source (nd-array): Image to which points belong.
        img_dest (nd-array): Destination image.
        
    Retruns:
        (nd-array): corresponding postions of points in form [[x1,y1], [x2,y2], ...] stacked vertically.
    """
    
    h_scale = img_dest.shape[0] / img_source.shape[0]
    w_scale = img_dest.shape[1] / img_source.shape[1]
    return (points * np.array([w_scale, h_scale]).reshape(1,2)).astype('int32')



class FaceExtractor:
    def __init__(self, minsize=20, threshold=[ 0.7, 0.7, 0.9], factor=0.71):
        with tf.Graph().as_default():
            self.sess = tf.Session()
            self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)
        self.mtcnn_img_size = (250,250)
        self.minsize = minsize # minimum size of face
        self.threshold = threshold  # three steps's threshold
        self.factor = factor # scale factor
    def extract_faces(self, img):
        """Detects faces in an image, and aligns them based on their eyes position.
        
        Args:
            img (nd-array): numpy array representing an image.
        
        Returns:
            (list): A list containing nd-arrays each representing an aligned face.
            
        """
        original_image = img.copy()
        height, width = original_image.shape[:2] # image shape has 3 dimensions
        
        # mtcnn requires images to be at the size of about 250*250
        resized_img = cv2.resize(img, self.mtcnn_img_size)
        
        # mtcnn returns 5 points for each face it detects in an array of 10 [x1,...,x5,y1,...,y5].T form
        # note that the points array is vertical
        # bounding_boxes is in the form of [x1,y1,x2,y2,conf] for each face stacked vertically in an nd-array
        bounding_boxes, points = detect_face.detect_face(resized_img, self.minsize, self.pnet, \
                                                         self.rnet, self.onet, self.threshold, self.factor)
        
        # we transform all the points to be in [x,y] format
        # list of five points for each face (the first two points are eyes)
        faces_xy = [face.reshape(2,5).T for face in points.T]
        
        # list of 2*2 arrays of form nd-array[[x1,y1], [x2,y2]] with points specifing coreners of rectangle
        faces_boxes = [box[:4].reshape(2,2) for box in bounding_boxes.astype('int32')]
        
        
        
        # angle of the slope of the line connecting two eyes for each face
        angle = [calculate_degree(face_points[0], face_points[1]) for face_points in faces_xy]
        
        
        # iterating through the faces aligning, rescaling and cropping
        cropped_faces = []
        for i in range(len(faces_boxes)):
            
            original_face_points = scale_points(faces_xy[i], resized_img, original_image)
            # getting rotation matrix around eyes_center without scaling
#             rotation_mat = cv2.getRotationMatrix2D(eyes_centers[i], angle[i], 1.0)
            
            # the box coordinates in the original image
            original_box = scale_points(faces_boxes[i], resized_img, original_image)
            # getting the list of center of the eyes for all faces
            eyes_center = (int(sum(original_face_points[:2,0])/2) , int(sum(original_face_points[:2,1])/2))
            original_rotation_mat = cv2.getRotationMatrix2D(eyes_center, angle[i], 1.0)
            
            


            # refer to affine transformation matrix definition 
            rotated_original_image = cv2.warpAffine(original_image, original_rotation_mat, (width,height))
            # scale the bounding box to include areas around the face
            scaled_box = scale_box(original_box, 1.2)
            nose_position_rotated = apply_affine(original_face_points[2,:], original_rotation_mat)[0]
            
            # Center the cropping box around nose in the rotated original image
            box_width = abs(scaled_box[0,0] - scaled_box[1,0])
            box_height = abs(scaled_box[0,1] - scaled_box[1,1])
            box_nose_centered = np.array([[nose_position_rotated[0]-box_width//2, nose_position_rotated[1]-box_height//2], \
                                          [nose_position_rotated[0]+box_width//2, nose_position_rotated[1]+box_height//2]])
            cropped_faces.append(safe_crop(rotated_original_image, scaled_box))
            
        return cropped_faces


