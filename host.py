from socket import *
from pickle import *
import numpy as np
from struct import *

# 1. Back to your original setup
global_model_weights = np.zeros(16384) 
NUM_Partticipants = int(input("number of participants pooling : "))
connected_workers = []

def send_weights(conn, weights):
    pickled_data = dumps(weights)
    header = pack('>I', len(pickled_data))
    conn.sendall(header + pickled_data)
# conn is like a pipe in sockets , using sendall function to send header here

serverSocket = socket(AF_INET, SOCK_STREAM)
# af inet is ipv4 , so if i connect this across multiple laptops , i will trype ipconfig in it and use ipv4 address , sock_stream defines me using tcp over udp
serverSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1) 
# this basically helped in removing a big error , as it made a already running port instantly available(not the default) for reuse
serverSocket.bind(('127.0.0.1', 12344))
# binds to a network 
serverSocket.listen(5)

serverSocket.setblocking(True) 
# basically wait at a worker to share file before proceedint to next one

print('host is ready')

for _ in range(NUM_Partticipants):
    conn, addr = serverSocket.accept()
    print(f"Hospital connected: {addr}")
    send_weights(conn, global_model_weights)
    connected_workers.append(conn)
# this step sends weights(global) to each participant , if i dont save conn , i cant recall it and access it in particular without a line of code
for round_num in range(10):
    updates = []
    for conn in connected_workers:
        raw_n = conn.recv(4)
        n_samples = int(raw_n.decode())
        # n was basically the number of files i was reciving , this hels in federated average
        header = conn.recv(4)
        size = unpack('>I', header)[0]
        data = bytearray()
        while len(data) < size:
            chuck = conn.recv(min(size - len(data), 4096))
            # min coz what if only 100 bytes remain? extracting 4096 bytes is to get in stream and take the remaining using minimum
            data.extend(chuck)
        updates.append((loads(data), n_samples))
        # updates is being appended by a tuple of gradients an n
    total_data = sum(u[1] for u in updates)
    # total n
    avg_gradient = np.zeros(16384)
    for grad, n in updates:
        avg_gradient += grad * (n / total_data)
    global_model_weights += avg_gradient
    print(f"Round {round_num + 1} complete.")
    for conn in connected_workers:
        send_weights(conn, avg_gradient)
print("finally , host has the final weights")
print(f"avg weights: {np.mean(global_model_weights):.6f}")
# this is average of all weights

