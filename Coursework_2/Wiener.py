from io import BytesIO
import numpy as np
import requests
from scipy.linalg import inv

# grab the data from the server
r = requests.get('https://4G10.cbl-cambridge.org/data.npz', stream=True)
data = np.load(BytesIO(r.raw.read()))

Neural_test = data['neural_test']
Neural_train = data['neural_train']
Hand_train = data['hand_train']

lag = 8

X_train_lags = np.zeros((lag * 162, 400, 16 + 1 - lag))
for k in range(400):
    for t in range(16 + 1 - lag):
        for l in range(lag):
            X_train_lags[(l*162):((l+1)*162), k, t] = Neural_train[:, k, t + lag - l - 1]

Hand_train_wien = Hand_train[:, :, (lag - 1):16]

X_lagsTX_lags = np.tensordot(X_train_lags, X_train_lags, axes=([1, 2], [1, 2]))
W_wien = inv(X_lagsTX_lags) @ np.tensordot(X_train_lags, Hand_train_wien, axes=([1, 2], [1, 2]))

X_test_lags = np.zeros((lag * 162, 100, 16 + 1 - lag))
for k in range(100):
    for t in range(16 + 1 - lag):
        for l in range(lag):
            X_test_lags[(l*162):((l+1)*162), k, t] = Neural_test[:, k, t + lag - l - 1]

V_predict = np.tensordot(W_wien, X_test_lags, axes=([0], [0]))
