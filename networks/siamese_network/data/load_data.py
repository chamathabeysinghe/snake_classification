import sys
import numpy as np
from scipy.misc import imread
from skimage import transform
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image

train_path = os.path.join('./','train')
validation_path = os.path.join('./', 'validation')
save_path = './'


def load_images(path,n=0):
    X=[]
    y = []
    #we load every alphabet seperately so we can isolate them later
    for breed_id in os.listdir(path):
        print("loading snake breed: " + breed_id)
        # lang_dict[alphabet] = [curr_y,None]
        # alphabet_path = os.path.join(path,alphabet)
        #every letter/category has it's own column in the array, so  load seperately
        breed_path = os.path.join(path,breed_id)
        for filename in os.listdir(breed_path):
            image_path = os.path.join(breed_path, filename)
            # image = imread(image_path)
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.asarray(image)
            image = transform.resize(image,(150,150))
            X.append(image)
            y.append(int(breed_id))

    y = np.vstack(y)
    X = np.stack(X)
    return X,y


X,y = load_images(train_path)
print(X.shape)
print(y.shape)

# with open(os.path.join(save_path,"train.pickle"), "wb") as f:
#     pickle.dump((X,y),f)
#
#
# X,y = load_images(validation_path)
# with open(os.path.join(save_path,"val.pickle"), "wb") as f:
#     pickle.dump((X,y),f)