from socket import *
from pickle import *
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

# X,XX,Y,YY are basically converted to numpy arrays and have an images tensor(X,XX) corresponding to its label(Y,YY)

# this is the actual ml model
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def train_step(X, y, weights, learning_rate=0.01):
    predictions = sigmoid(np.dot(X, weights))
    error = predictions - y
    gradient = np.dot(X.T, error) / len(y)
    return weights - learning_rate * gradient

master_ip = '127.0.0.1' 
master_port = 12344

worker_socket = socket(AF_INET, SOCK_STREAM)
worker_socket.connect((master_ip, master_port))

n_value = str(len(Y)).zfill(4)
# its calculating n by seeing the number f labels 
header = worker_socket.recv(4)
size = unpack('>I', header)[0]
# the first thing i am gonna recieve is info on number of bytes that are gonnA come
data = bytearray()
while len(data) < size:
    data.extend(worker_socket.recv(min(size - len(data), 4096)))
global_weights = loads(data)
# this accepts info on global weights from the host

# ml maths part
initial_raw = sigmoid(np.dot(XX, global_weights)) 
initial_bin = (initial_raw > 0.5).astype(float)
initial_acc = np.mean(initial_bin == YY)

accuracies = [initial_acc * 100]
for round_num in range(10):
    worker_socket.sendall(n_value.encode())
    current_weights = global_weights.copy()
    # avoids overwriting my initial data
    for _ in range(10):
        current_weights = train_step(X, Y, current_weights)
    delta = (current_weights - global_weights) + np.random.normal(0, 0.01, global_weights.shape)
    pickled = dumps(delta)
    # pickling data cnverts it into a stream of bytes
    worker_socket.sendall(pack('>I', len(pickled)) + pickled)
    header = worker_socket.recv(4)
    size = unpack('>I', header)[0]
    data = bytearray()
    while len(data) < size:
        data.extend(worker_socket.recv(min(size - len(data), 4096)))
    global_weights += loads(data)
    raw_preds = sigmoid(np.dot(XX, global_weights))
    bin_preds = (raw_preds > 0.5).astype(float)
    round_acc = np.mean(bin_preds == YY) * 100
    accuracies.append(round_acc)
worker_socket.close()
print("Success")


# ml maths again
raw_predictions = sigmoid(np.dot(XX, global_weights))
binary_predictions = (raw_predictions > 0.5).astype(float)
final_test_accuracy = np.mean(binary_predictions == YY)

print("Final results")
print(f"Total Test Images: {len(YY)}")
print(f"Correct Matches:      {int(np.sum(binary_predictions == YY))}")
print(f"Final Accuracy:    {final_test_accuracy * 100:.2f}%")
print(f"Initial accuracy : {initial_acc*100:.2f}%")



import matplotlib.pyplot as plt
plt.plot(range(len(accuracies)), accuracies, marker='o', color='b', label='Test Accuracy')
plt.title('Federated Learning Progress (Brain Tumor Detection)')
plt.xlabel('Communication Rounds')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.show()