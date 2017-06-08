import csv
import numpy as np
from sklearn import manifold, decomposition, cluster

import plot
import utils

# plt.rcParams['image.cmap'] = 
with open('data/MDW.csv', 'rb') as MDWcsv:
	# np.set_printoptions(threshold=np.inf, suppress=True)
	csv = csv.reader(MDWcsv, delimiter=',')
	data = utils.format_data(csv)[0]

	print("Using PCA, n=2")
	pca = decomposition.PCA(n_components=2)
	output = pca.fit_transform(data)
	plot.plot2d(output[::7], "humidity", color=data[::7,7])
	plot.plot2d(output[::7], "temperature", color=data[::7,1])

	print("Using PCA, n=3")
	pca = decomposition.PCA(n_components=3)
	output = pca.fit_transform(data)
	plot.plot3d(output[::7], "humidity3d", color=data[::7,7])
	plot.plot3d(output[::7], "temperature3d", color=data[::7,1])

