## This file allows online augmentation of face photos in different backgournds with DLIB face 
## Extraction

import numpy as np
import imutils
import dlib
import cv2
import os
from imutils import face_utils
from PIL import Image
import random
import matplotlib.pyplot as plt

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# load the input image, resize it, and convert it to grayscale


def get_face_points(img, width=500):
    image = img
    image = imutils.resize(image, width)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    
    if not rects:
#         plt.imshow(img)
#         print('8'*10)
        return None
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    return shape




def get_transparent_cropped(img,poly):
    "crops and makes the rest transparent with poly"
    hull = poly
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
#     plt.imshow(img)

    pts = np.array(hull)

    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)


    return dst



def merge(big_img, small_img, x_offset, y_offset):
    'returns merged image and box around small image mereged with big'
    s_img = small_img.copy()
    s_img = cv2.cvtColor(s_img , cv2.COLOR_RGB2RGBA)
    l_img = cv2.cvtColor(big_img , cv2.COLOR_RGB2RGBA)

    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    

    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                  alpha_l * l_img[y1:y2, x1:x2, c])
        
    return l_img , [(x1,y1), (x2,y2)]



def get_face_foreground(img,dest_width=200):
    img = imutils.resize(img, width=500)

    s = get_face_points(img, width=500)
    if s is None:
        return 
    
    hull = np.squeeze(cv2.convexHull(s, returnPoints = True))
    face = get_transparent_cropped(img, hull)
    return imutils.resize(face, dest_width)





def get_bg_images(directory, width=500):
    'returns resized backgrounds'
    background_images = []
    for file in os.listdir(directory):
        im = cv2.imread(directory + '/' + file)
        if im is None:
            print(file)
        background_images.append(im)

    bg = []
    for image in background_images:
        bg.append(imutils.resize(image, width))
    return bg





def get_foreground_images(directory):
    fg = []
    for face_image in os.listdir(directory):
        im = cv2.imread(directory + '/' + face_image)
        # quick and diryt way to tackle with weired images
        try:
            face = get_face_foreground(im)
            if face is not None:
                fg.append(face)
        except:
            continue
    return fg





def merge_random_place(big, small):
    xmax = big.shape[1] - small.shape[1]
    ymax = big.shape[0] - small.shape[0]
    if ymax > 0 and xmax > 0:
    
        m, box = merge(big, small , random.randint(0,xmax),random.randint(0,ymax))
        return m, box
    else:
        return None




def random_resize(img,min_width=75, max_width=250):
    return imutils.resize(img, width=random.randint(min_width, max_width))




def augment(fore_ground_path, background_path, num=200):
    
    backgrounds = get_bg_images(background_path)
#     print(len(backgrounds))
    fgs = get_foreground_images(fore_ground_path)
    
    # add original foreground images without cleaning to backgrounds
    original_face_pics = []
    for face_image in os.listdir(fore_ground_path):
        im = cv2.imread(fore_ground_path + '/' + face_image)
        original_face_pics.append(im)
    cropped_face_pics = get_foreground_images(fore_ground_path)
    
    all_faces = original_face_pics + cropped_face_pics
    

    
    augmented = []
    boxes = []
    while len(augmented) < num:
        fore = random_resize(random.choice(all_faces))
        back = random.choice(backgrounds)
        

        image_box = merge_random_place(back, fore)
        if image_box:
            res_image, res_box  = image_box
            augmented.append(res_image)
            boxes.append(res_box)
    pil_aug = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in augmented]
    pil_aug = [Image.fromarray(img) for img in pil_aug]
    return pil_aug , boxes




