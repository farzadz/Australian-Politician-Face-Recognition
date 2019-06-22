import os
import logging
from extractor import FaceExtractor 
import matplotlib.pyplot as plt
import sys


directory = sys.argv[1]


logger = logging.getLogger(__name__)
logger.setLevel('INFO')
f_handler = logging.FileHandler('extraction.log')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.addHandler(f_handler)

def extract_faces_from_dir(directory):
    
    extractor = FaceExtractor()
    
    if not os.path.exists('./faces'):
        os.makedirs('./faces')
        
    # dangerous!
    extracted_dir = os.path.join('./faces', directory.split('/')[-1])

    
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir)
    for parent ,__, files in os.walk(directory):
        num_faces = 0
        for file in files:
            try:
                img = plt.imread(os.path.join(parent,file))

                faces = extractor.extract_faces(img)
                
                num_faces += len(faces)
                for i, face in enumerate(faces):
                    plt.imsave(os.path.join(extracted_dir, file.split('.')[-2] + '-face-' + str(i)) + ".jpg", face)
            except Exception as e:
                logger.error("Failed extraction for file: {}, {}".format(file, e))
    logger.info("Extracted {} faces from {} directory".format(num_faces, parent))


    
extract_faces_from_dir(directory)