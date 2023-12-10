import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data_eng import data_preprocessing

def standard_normalize(x):
	x_mean = np.mean(x, axis=1)[:, np.newaxis]
	x_std = np.std(x, axis=1)[:, np.newaxis]
	# exploding exponential checks
	idx = np.argwhere(x_std < 1e-2)
	if len(idx) > 0:
		ind = [v[0] for v in idx]
		adjust = np.zeros((x_std.shape))
		adjust[ind] = 1.0
		x_std += adjust
	return (x - x_mean) / x_std

def compute_pcs(X, lam):
	P = float(X.shape[1])
	Cov = 1/P * np.dot(X, X.T) + lam*np.eye(X.shape[0])
	D, V = np.linalg.eigh(Cov)
	return D, V

def pca_sphere(x):
	# compute eigenvalues and eigenvectors of covariance matrix
	x = standard_normalize(x)
	D, V = compute_pcs(x, 1e-5)
	D = np.abs(D)
	# compute the projection matrix
	P = np.dot(V.T, x)
	return P/(D[:,np.newaxis]**0.5)

def load_data(standard_normalized=False, pca_sphered=False, train=True):
	if train:
		x, y = data_preprocessing()
	else:
		x, y = data_preprocessing(train=False, file='test.csv')
	if standard_normalized:
		x = standard_normalize(x)
	if pca_sphered:
		x = standard_normalize(x)
		x = pca_sphere(x)

	return x, y
	

if __name__ == "__main__":
	x, y = load_data()
	print(x.shape)
	print(y.shape)