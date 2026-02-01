from socket import *
import numpy as np
from struct import *
import zipfile
import os
import random
from PIL import Image 
def transformations(img_path):
    img = Image.open(img_path).convert('L') 
    img = img.resize((128, 128))        
    arr = np.array(img).astype(np.float32)
    arr = (arr / 127.5) - 1.0             
    return arr.flatten()          
#converting l changes to to8 bit pixels,then define resolutions , the change pixel value to be bounded in [-1,1], then return a 1d array
# tho this  might seem bad but pil interpolation actually is comparitive around neighboring pixels to lower the particular one   
print("client 1 is connected")
zip_file_name = r"C:\Users\GAURAV\OneDrive\Desktop\data_sharing\archive.zip"
extract_dir = 'brain_tumor_data'
if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
# this will create a new directory of images in my ide

image_yes = os.listdir("brain_tumor_data/Training/glioma")
image_no = os.listdir("brain_tumor_data/Testing/notumor")
# this will access them

yes_train = image_yes[:150]
no_train = image_no[:150]
# i am defining it to train on 300 images , 50-50 for yes/no
compiled = []
for img in yes_train:
    compiled.append((os.path.join("brain_tumor_data/Training/glioma", img), 1))
for img in no_train:
    compiled.append((os.path.join("brain_tumor_data/Testing/notumor", img), 0))
# os.join is bacially connecting the directory to image

random.shuffle(compiled)
# testing data is on 100 images
yes_test = image_yes[300:350]
no_test = image_no[300:350]
compiled_test = []
for imgs in yes_test:
    compiled_test.append((os.path.join("brain_tumor_data/Training/glioma", imgs), 1))
for imgs in no_test:
    compiled_test.append((os.path.join("brain_tumor_data/Testing/notumor", imgs), 0))
random.shuffle(compiled_test)
XX, YY = [], []
for i, (path, label) in enumerate(compiled_test):
    try:
        XX.append(transformations(path)) 
        YY.append(label)
    except:
        pass
XX = np.array(XX)
YY= np.array(YY).astype(float)

print("images loading", flush=True)
X, Y = [], []
for i, (path, label) in enumerate(compiled):
    try:
        X.append(transformations(path)) 
        Y.append(label)
    except:
        pass

X = np.array(X)
Y = np.array(Y).astype(float)