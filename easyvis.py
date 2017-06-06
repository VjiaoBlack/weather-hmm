import csv
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, cluster)
from sklearn.cross_decomposition import CCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from calendar import monthrange, isleap
import datetime as dt


def update(i):
	label = 'timestep {0}'.format(i)
	# "artists" that have to be redrawn for this frame.
	ax.view_init(30, 60.0 + i * 2.5)
	return plt, ax

# plt.rcParams['image.cmap'] = 
with open('data/SFO.csv', 'rb') as MDWcsv:
	# np.set_printoptions(threshold=np.inf, suppress=True)
	np.set_printoptions(threshold=np.inf, suppress=True, precision=3)
	MDWdata_orig = csv.reader(MDWcsv, delimiter=',')
	MDWdata_orig = np.asarray(list(MDWdata_orig))
	# If data has "min / max / mean", just keep mean.
	MDWdata_orig = np.delete(MDWdata_orig, [18, 21] ,1)
	MDWdata_orig = MDWdata_orig[np.all(MDWdata_orig != '',axis=1)]
	MDWdata_orig = MDWdata_orig[1:]
	MDWdata = MDWdata_orig
	MDWdata[MDWdata == 'T'] = 0.001 # T is when there is rain but not enough to be measured


	# data = np.concatenate((data,np.zeros((len(data), 1))), axis=1)
	# for i in range(0, len(data)):
	# 	angle = np.radians(float(data[i][19]))
	# 	data[i][19] = np.cos(angle) * 100.0
	# 	data[i][20] = np.sin(angle) * 100.0



	cmap = matplotlib.cm.get_cmap('inferno')

	date_vector = np.zeros((len(MDWdata), 1))
	for i in range(0, len(MDWdata)):
		date = map(int,MDWdata[i][0].split("-"))
		# MDWdata[i][0] = float(date_tup[0]) + float()
		days = (dt.date(date[0], date[1], date[2]) - dt.date(date[0],1,1)).days
		
		if isleap(date[0]): # float(date[0]) +
			metric = (days / 366.0) 
		else:
			metric = (days / 365.0)

		date_vector[i] = metric * 12.0 + 1.0
		MDWdata[i][0] = date_vector[i][0]

	data = MDWdata.astype(np.float)



	cmap = matplotlib.cm.get_cmap('inferno')



	# CST, MaxTemperatureF, MeanTemperatureF, MinTemperatureF, MaxDewPointF, MeanDewPointF, MinDewpointF, MaxHumidity, MeanHumidity, MinHumidity, MaxSeaLevelPressureIn, MeanSeaLevelPressureIn, MinSeaLevelPressureIn, MaxVisibilityMiles, MeanVisibilityMiles, MinVisibilityMiles, MaxWindSpeedMPH, MeanWindSpeedMPH, MaxGustSpeedMPH, PrecipitationIn, CloudCover, Events, WindDirDegrees
	plt.figure(figsize=(8,5), dpi=200)	
	plt.scatter(date_vector, data[:,1], s=0.5, linewidth=1, c="black", alpha=0.2) # color is defined by 8th param
	plt.show()
	title = "SFO-time-temp.png"
	if title is not None:
		 plt.title(title)
	plt.savefig(title)
	plt.close()

	plt.figure(figsize=(8,5), dpi=200)
	plt.scatter(date_vector, data[:,7], s=0.5, linewidth=1, c="black", alpha=0.2) # color is defined by 1st param
	plt.show()
	title = "SFO-time-humidity.png"
	if title is not None:
		 plt.title(title)
	plt.savefig(title)
	plt.close()

	plt.figure(figsize=(8,5), dpi=200)
	plt.scatter(date_vector, data[:,20], s=0.5, linewidth=1, c="black", alpha=0.2) # color is defined by 1st param
	plt.show()
	title = "SFO-time-winddir.png"
	if title is not None:
		 plt.title(title)
	plt.savefig(title)
	plt.close()

	plt.figure(figsize=(8,5), dpi=200)
	plt.scatter(date_vector, data[:,20], s=0.5, linewidth=1, cmap=cmap, c=data[:,7]) # color is defined by 1st param
	plt.show()
	title = "SFO-time-winddir-by-moisture.png"
	if title is not None:
		 plt.title(title)
	plt.savefig(title)
	plt.close()







