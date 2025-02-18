import os
from openimages.download import download_dataset
import shutil

# initialize global variables
data_dir = "./images"
number_for_samples = 334
classes = ["Goose", "Jellyfish", "Snail"] 

# check if the directories and images exist, if yes delete them
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
    os.makedirs(data_dir)
else:
    os.makedirs(data_dir)

# download dataset from openImages
download_dataset(data_dir, classes, limit=number_for_samples)