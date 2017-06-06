import csv
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, cluster)
from sklearn.cross_decomposition import CCA
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

# plt.rcParams['image.cmap'] = 
with open('data/MDW.csv', 'rb') as MDWcsv:
	# np.set_printoptions(threshold=np.inf, suppress=True)
	MDWdata = csv.reader(MDWcsv, delimiter=',')
	MDWdata = np.asarray(list(MDWdata))
	MDWdata = MDWdata[1:]
	MDWdata = np.delete(MDWdata, [0, 18, 21] ,1)
	MDWdata[MDWdata == 'T'] = 0.01 # T is when there is rain but not enough to be measured
	MDWdata = MDWdata[np.all(MDWdata != '',axis=1)]
	# MDWdata = np.concatenate((np.full((1,4000),0).T, MDWdata.astype(np.float)[0:4000:]),axis=1)
	data = MDWdata.astype(np.float)

	print(data.shape)

	print("Using PCA")
	pca = decomposition.PCA(n_components=2)

	train = pca.fit_transform(data[:,1:])

	emb = train[::]#[0:1080:]
	label = data[::]#[0:1080:]

	print(emb.shape)

	cmap = matplotlib.cm.get_cmap('inferno')


	# change degree value to an actually meaningful value... ugh
	data = np.concatenate((data,np.zeros((len(data), 1))), axis=1)
	for i in range(0, len(data)):
		angle = np.radians(float(data[i][19]))
		data[i][19] = np.cos(angle) * data[i][16]
		data[i][20] = np.sin(angle) * data[i][16]

	# CST, MaxTemperatureF, MeanTemperatureF, MinTemperatureF, MaxDewPointF, MeanDewPointF, MinDewpointF, MaxHumidity, MeanHumidity, MinHumidity, MaxSeaLevelPressureIn, MeanSeaLevelPressureIn, MinSeaLevelPressureIn, MaxVisibilityMiles, MeanVisibilityMiles, MinVisibilityMiles, MaxWindSpeedMPH, MeanWindSpeedMPH, MaxGustSpeedMPH, PrecipitationIn, CloudCover, Events, WindDirDegrees


	plt.figure(figsize=(8,5), dpi=200)	
	plt.scatter(emb[:,0], emb[:,1], s=0.5, linewidth=1, c=cmap((label[:,7] - np.min(label[:,7])) / (np.max(label[:,7]) - np.min(label[:,7]))), alpha=0.5) # color is defined by 7th param
	plt.show()
	title = "humidity-scaledwind.png"
	if title is not None:
		 plt.title(title)
	plt.savefig(title)
	plt.close()

	plt.figure(figsize=(8,5), dpi=200)
	plt.scatter(emb[:,0], emb[:,1], s=0.5, linewidth=1, c=cmap((label[:,1] - np.min(label[:,1])) / (np.max(label[:,1]) - np.min(label[:,1]))), alpha=0.5) # color is defined by 1st param
	plt.show()
	title = "temperature-scaledwind.png"
	if title is not None:
		 plt.title(title)
	plt.savefig(title)
	plt.close()


	print("Using PCA3")
	pca = decomposition.PCA(n_components=3)

	train = pca.fit_transform(data[:,1:])

	emb = train[::]#[0:1080:]
	label = data[::]#[0:1080:]

	print(emb.shape)

	fig = plt.figure(figsize=(8,5), dpi=200)	
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(30,60)
	ax.scatter(xs=emb[:,0], ys=emb[:,1], zs=emb[:,2], s=0.5, linewidth=1, c=cmap((label[:,7] - np.min(label[:,7])) / (np.max(label[:,7]) - np.min(label[:,7]))), alpha=0.5) # color is defined by 7th param
	plt.show()
	title = "humidity3d-scaledwind.png"
	if title is not None:
		 plt.title(title)
	plt.tight_layout()
	plt.savefig(title)
	plt.close()

	fig = plt.figure(figsize=(8,5), dpi=200)
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(30,60)
	ax.scatter(xs=emb[:,0], ys=emb[:,1], zs=emb[:,2], s=0.5, linewidth=1, c=cmap((label[:,1] - np.min(label[:,1])) / (np.max(label[:,1]) - np.min(label[:,1]))), alpha=0.5) # color is defined by 1st param
	plt.show()
	title = "temperature3d-scaledwind.png"
	if title is not None:
		 plt.title(title)
	plt.tight_layout()
	plt.savefig(title)
	plt.close()
