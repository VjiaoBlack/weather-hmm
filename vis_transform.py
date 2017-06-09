import csv
import numpy as np
from sklearn import manifold, decomposition, cluster

import plot
import utils

# Only looks at MDW cuz this is slow, 
# and MDW offers enough interesting things to see (or lack thereof)
with open('data/MDW.csv', 'rb') as MDWcsv:
	# np.set_printoptions(threshold=np.inf, suppress=True)
	csv = csv.reader(MDWcsv, delimiter=',')
	data = utils.format_data(csv)[0]

	print("Using PCA, n=2")
	pca = decomposition.PCA(n_components=2)
	output = pca.fit_transform(data)
	plot.plot2d(tuple(output[::7].T), "humidity-pca", c=data[::7,7], a=0.8)
	plot.plot2d(tuple(output[::7].T), "temperature-pca", c=data[::7,1], a=0.8)

	print("Using PCA, n=3")
	pca = decomposition.PCA(n_components=3)
	output = pca.fit_transform(data)
	plot.plot3d_anim(tuple(output[::7].T), "humidity-pca3d", c=data[::7,7], a=0.8)
	plot.plot3d_anim(tuple(output[::7].T), "temperature-pca3d", c=data[::7,1], a=0.8)

	print("Using tSNE, n=2, p=100, training on [::7]")
	tsne = manifold.TSNE(n_components=2, perplexity=100)
	output = tsne.fit_transform(data[::7])
	plot.plot2d(tuple(output.T), "humidity-100tsne", c=data[::7,7], a=0.8)
	plot.plot2d(tuple(output.T), "temperature-100tsne", c=data[::7,1], a=0.8)

	print("Using tSNE, n=3, p=100, training on [::7]")
	tsne = manifold.TSNE(n_components=3, perplexity=100)
	output = tsne.fit_transform(data[::7])
	plot.plot3d_anim(tuple(output.T), "humidity-100tsne3d", c=data[::7,7], a=0.8)
	plot.plot3d_anim(tuple(output.T), "temperature-100tsne3d", c=data[::7,1], a=0.8)

	print("Using tSNE, n=2, p=15, training on [::7]")
	tsne = manifold.TSNE(n_components=2, perplexity=15)
	output = tsne.fit_transform(data[::7])
	plot.plot2d(tuple(output.T), "humidity-15tsne", c=data[::7,7], a=0.8)
	plot.plot2d(tuple(output.T), "temperature-15tsne", c=data[::7,1], a=0.8)

	print("Using tSNE, n=3, p=15, training on [::7]")
	tsne = manifold.TSNE(n_components=3, perplexity=15)
	output = tsne.fit_transform(data[::7])
	plot.plot3d_anim(tuple(output.T), "humidity-15tsne3d", c=data[::7,7], a=0.8)
	plot.plot3d_anim(tuple(output.T), "temperature-15tsne3d", c=data[::7,1], a=0.8)