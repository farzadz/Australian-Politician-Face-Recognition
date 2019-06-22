import os
import logging
import Clustering
import matplotlib.pyplot as plt
import sys


directory = sys.argv[1]


logger = logging.getLogger(__name__)
logger.setLevel('INFO')
f_handler = logging.FileHandler('clustering.log')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.addHandler(f_handler)

prune_faces_from_dir(directory)