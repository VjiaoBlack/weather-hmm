import csv
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, cluster)
from sklearn.cross_decomposition import CCA


# plt.rcParams['image.cmap'] = 
with open('data/MDW.csv', 'rb') as MDWcsv:
	# np.set_printoptions(threshold=np.inf, suppress=True)
	MDWdata = csv.reader(MDWcsv, delimiter=',')
	MDWdata = np.asarray(list(MDWdata))
	MDWdata = MDWdata[1:,1:-2]
	MDWdata = MDWdata[np.all(MDWdata != '',axis=1)]
	MDWdata[MDWdata == 'T'] = 0.01 # T is when there is rain but not enough to be measured
	MDWdata = np.concatenate((np.full((1,4000),0).T, MDWdata.astype(np.float)[0:4000:]),axis=1)

	with open('data/LGA.csv', 'rb') as LGAcsv:
		LGAdata = csv.reader(LGAcsv, delimiter=',')
		LGAdata = np.asarray(list(LGAdata))
		LGAdata = LGAdata[1:,1:-2]
		LGAdata = LGAdata[np.all(LGAdata != '',axis=1)]
		LGAdata[LGAdata == 'T'] = 0.01 # T is when there is rain but not enough to be measured
		LGAdata = np.concatenate((np.full((1,4000),0.33).T, LGAdata.astype(np.float)[0:4000:]),axis=1)

		with open('data/ORD.csv', 'rb') as ORDcsv:
			ORDdata = csv.reader(ORDcsv, delimiter=',')
			ORDdata = np.asarray(list(ORDdata))
			ORDdata = ORDdata[1:,1:-2]
			ORDdata = ORDdata[np.all(ORDdata != '',axis=1)]
			ORDdata[ORDdata == 'T'] = 0.01 # T is when there is rain but not enough to be measured
			ORDdata = np.concatenate((np.full((1,4000),0.67).T, ORDdata.astype(np.float)[0:4000:]),axis=1)

			with open('data/SFO.csv', 'rb') as SFOcsv:
				SFOdata = csv.reader(SFOcsv, delimiter=',')
				SFOdata = np.asarray(list(SFOdata))
				SFOdata = SFOdata[1:,1:-2]
				SFOdata = SFOdata[np.all(SFOdata != '',axis=1)]
				SFOdata[SFOdata == 'T'] = 0.01 # T is when there is rain but not enough to be measured
				SFOdata = np.concatenate((np.full((1,4000),1.0).T, SFOdata.astype(np.float)[0:4000:]),axis=1)

				data = np.concatenate((MDWdata, LGAdata, ORDdata, SFOdata))

				print(data.shape)

				print("Using PCA")
				pca = decomposition.PCA(n_components=2)

				train = pca.fit_transform(data[:,1:])

				emb = train[::]#[0:1080:]
				label = data[::]#[0:1080:]

				print(emb.shape)

				# CST, MaxTemperatureF, MeanTemperatureF, MinTemperatureF, MaxDewPointF, MeanDewPointF, MinDewpointF, MaxHumidity, MeanHumidity, MinHumidity, MaxSeaLevelPressureIn, MeanSeaLevelPressureIn, MinSeaLevelPressureIn, MaxVisibilityMiles, MeanVisibilityMiles, MinVisibilityMiles, MaxWindSpeedMPH, MeanWindSpeedMPH, MaxGustSpeedMPH, PrecipitationIn, CloudCover, Events, WindDirDegrees
				plt.figure(figsize=(8,5), dpi=200)	
				plt.scatter(emb[:,0], emb[:,1], s=0.5, linewidth=1, c=label[:,8]) # color is defined by 8th param
				plt.show()
				title = "overall-humidity.png"
				if title is not None:
					 plt.title(title)
				plt.savefig(title)
				plt.close()

				plt.figure(figsize=(8,5), dpi=200)
				plt.scatter(emb[:,0], emb[:,1], s=0.5, linewidth=1, c=label[:,2]) # color is defined by 1st param
				plt.show()
				title = "overall-temperature.png"
				if title is not None:
					 plt.title(title)
				plt.savefig(title)
				plt.close()

				plt.figure(figsize=(8,5), dpi=200)
				plt.scatter(emb[:,0], emb[:,1], s=0.5, linewidth=1, c=label[:,0]) # color is defined by 1st param
				plt.show()
				title = "overall.png"
				if title is not None:
					 plt.title(title)
				plt.savefig(title)
				plt.close()