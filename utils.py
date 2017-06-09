import csv
import numpy as np
from calendar import monthrange, isleap
import datetime as dt

def format_data(csv_data):
	data = np.asarray(list(csv_data))

	# remove header row
	data = data[1:]

	# processing dates separately
	# dates[:,0] 	int part - year, frac part - day / total days
	# dates[:,1:3]	converts date to a unit vector
	dates = np.zeros((len(data), 3))
	for i in range(0, len(dates)):
		date = map(int,data[i][0].split("-"))
		days = (dt.date(date[0], date[1], date[2]) - dt.date(date[0],1,1)).days
		
		if isleap(date[0]): # float(date[0]) +
			metric = (days / 366.0) 
		else:
			metric = (days / 365.0)

		dates[i][0] = metric
		dates[i][1] = np.cos(metric * np.pi * 2.0)
		dates[i][2] = np.sin(metric * np.pi * 2.0)

	# separate dates from data
	dates = dates.astype(np.float)

	# remove time, gust speed, events
	data = np.delete(data, [0, 18, 21], 1); 

	# T is when there is rain, but not enough to be measured
	data[data == 'T'] = 0.01

	# if we get null values for clouds or rain, assume it's 0
	# Note: last 2 values in array should be rain
	data[:,-2:][data[:,-2:] == ''] = 0

	# I don't have a good enough way to fill in these data points
	# so I decide to delete incomplete data entries.
	# Note: this /might/ mess with our model slightly, since
	# our HMM models assume our data is taken daily. 
	dates = dates[np.all(data != '', axis=1)]
	data = data[np.all(data != '', axis=1)]

	# change wind degree to a wind-speed weighted vector
	wind = data[:,19]
	data = np.concatenate((data,np.zeros((len(data), 1))), axis=1)
	for i in range(0, len(data)):
		angle = np.radians(float(data[i][19]))
		data[i][19] = np.cos(angle) * float(data[i][16])
		data[i][20] = np.sin(angle) * float(data[i][16])

	# convert everything to floats
	data = data.astype(np.float)

	return (data,dates,wind)
