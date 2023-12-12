from io import BytesIO
import numpy as np
import requests
from scipy.ndimage import gaussian_filter
from scipy.linalg import inv

# grab the data from the server
r = requests.get('https://4G10.cbl-cambridge.org/data.npz', stream=True)
data = np.load(BytesIO(r.raw.read()))

Neural_test = data['neural_test']
Neural_train = data['neural_train']
Hand_train = data['hand_train']


# Gaussian Smoothing
def smoother(Neural_train, std):
    return gaussian_filter(Neural_train, sigma=std, axes=2)


# Decoder
def W_MAP_calc(V, X_til, lam):
    lam_I_n = lam * np.identity(X_til.shape[0])
    brackets = inv(np.tensordot(X_til, X_til, axes=([1, 2], [1, 2])) + lam_I_n)
    return np.tensordot(V, X_til, axes=([1, 2], [1, 2])) @ brackets


# Predict
for std in [0.88]:
    Neural_train_filt = smoother(Neural_train, std)
    Neural_test_filt = smoother(Neural_test, std)
    for lam in [0.01, 0.1, 1, 2.5, 5]:
        W_MAP = W_MAP_calc(Hand_train, Neural_train, lam)
        Predicted_vel = np.tensordot(W_MAP, Neural_test_filt, axes=([1], [0]))
