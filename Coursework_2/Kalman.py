from io import BytesIO
import numpy as np
import requests
from scipy.linalg import inv

# grab the data from the server
r = requests.get('https://4G10.cbl-cambridge.org/data.npz', stream=True)
data = np.load(BytesIO(r.raw.read()))


# 3.1 An autoregressive prior for hand kinematics
z_0 = np.random.multivariate_normal(data['hand_KF_mu0'].reshape((-1,)), data['hand_KF_Sigma0'], size=(100,))

epsilon = np.random.multivariate_normal(np.zeros(10), data['hand_KF_Q'], size=(100, 16))

A = data['hand_KF_A']
C = data['hand_KF_C']
R = data['hand_KF_R']
Q = data['hand_KF_Q']
Sigma0 = data['hand_KF_Sigma0']
mu0 = data['hand_KF_mu0'].reshape((10,))
v_train = data['hand_train']

Sigma_inv = np.zeros((10, 10, 400, 16))
mu = np.zeros((10, 400, 16))

CTRC = C.T @ inv(R) @ C
for i in range(Sigma_inv.shape[2]):
    P = inv(A @ Sigma0 @ A.T + Q)
    Sigma_inv[:, :, i, 0] = CTRC + P
    mu[:, i, 0] = inv(Sigma_inv[:, :, i, 0]) @ (P @ A @ mu0 + C.T @ inv(R) @ v_train[:, i, 0])
    for j in range(Sigma_inv.shape[3] - 1):
        P = inv(A @ inv(Sigma_inv[:, :, i, j]) @ A.T + Q)
        Sigma_inv[:, :, i, j + 1] = CTRC + inv(A @ inv(Sigma_inv[:, :, i, j]) @ A.T + Q)
        mu[:, i, j + 1] = inv(Sigma_inv[:, :, i, j + 1]) @ (P @ A @ mu[:, i, j] + C.T @ inv(R) @ v_train[:, i, j + 1])

Sigma_tilde = np.zeros((10, 10, 400, 16))
mu_tilde = np.zeros((10, 400, 16))

for i in range(Sigma_tilde.shape[2]):
    mu_tilde[:, : i, -1] = mu[:, : i, -1]
    Sigma_tilde[:, :, i, -1] = inv(Sigma_inv[:, :, i, -1])
    for j in range(Sigma_tilde.shape[3] - 1)[::-1]:
        Sigma_t = inv(Sigma_inv[:, :, i, j])
        P_t = A @ Sigma_t @ A.T + Q
        G_t = Sigma_t @ A.T @ inv(P_t)
        mu_tilde[:, i, j] = mu[:, i, j] + G_t @ (mu_tilde[:, i, j+1] - A @ mu[:, i, j])
        Sigma_tilde[:, :, i, j] = Sigma_t + G_t @ (Sigma_tilde[:, :, i, j+1] - P_t) @ G_t.T


# 3.2 Building an LDS model of neural data using supervised learning


def center_data(data):
    mean = data.mean(axis=(1, 2))
    mean = mean.reshape(162, 1, 1)
    data_centered = data - mean
    return data_centered


neural_train_cent = center_data(data['neural_train'])
neural_test_cent = center_data(data['neural_test'])

D_opt = np.tensordot(neural_train_cent, mu_tilde, axes=([1, 2], [1, 2])) @ inv(np.tensordot(mu_tilde, mu_tilde, axes=([1, 2], [1, 2])))

S_opt = (np.tensordot(neural_train_cent, neural_train_cent, axes=([1, 2], [1, 2])) - D_opt @ np.tensordot(mu_tilde, neural_train_cent, axes=([1, 2], [1, 2])))/(400*16)

# 3.3 Using Kalman filtering to predict the hand velocity
Sigma_inv_test = np.zeros((10, 10, 100, 16))
mu_test = np.zeros((10, 100, 16))

S_inv = inv(S_opt)
DTSD = D_opt.T @ S_inv @ D_opt
for i in range(Sigma_inv_test.shape[2]):
    P = inv(A @ Sigma0 @ A.T + Q)
    Sigma_inv_test[:, :, i, 0] = DTSD + P
    mu_test[:, i, 0] = inv(Sigma_inv_test[:, :, i, 0]) @ (P @ A @ mu0 + D_opt.T @ S_inv @ neural_test_cent[:, i, 0])
    for j in range(Sigma_inv_test.shape[3] - 1):
        P = inv(A @ inv(Sigma_inv_test[:, :, i, j]) @ A.T + Q)
        Sigma_inv_test[:, :, i, j + 1] = DTSD + inv(A @ inv(Sigma_inv_test[:, :, i, j]) @ A.T + Q)
        mu_test[:, i, j + 1] = inv(Sigma_inv_test[:, :, i, j + 1]) @ (P @ A @ mu_test[:, i, j] + D_opt.T @ S_inv @ neural_test_cent[:, i, j + 1])

v_test = np.tensordot(C, mu_test, axes=([1], [0]))
