import warnings
import csv
import numpy as np
from sklearn import decomposition
from hmmlearn import hmm

import plot
import utils

DATASETS = ["LGA", "SFO", "MDW", "ORD"]
HMM_CLASSES = [2, 4, 6, 8, 10]
NUM_TRAIN = 20000

np.set_printoptions(formatter={'float': '{:05.2f}'.format})

# This file does the testing without clearly time-dependent variables,
# such as wind direction, temperature, dewpoint, etc.
# 
# It would thus not make sense to /try/ to predict temperature, etc. 
# from our HMM classes (since we are ignoring them)

warnings.filterwarnings("ignore")
for DATASET in DATASETS:
	with open("data/" + DATASET + ".csv", 'rb') as rawcsv:

		our_csv = csv.reader(rawcsv, delimiter=',')
		data, dates, wind = utils.format_data(our_csv)
		orig_data = np.array(data)

		data = data[:,6:-2]

		print("\n#####  " + DATASET + "  #####")
		print(data.shape)
		print

		# overall standard deviations of data
		std = np.std(data, axis=0)

		# HMM class estimates
		train = data[0:NUM_TRAIN].astype(int)
		test  = data[NUM_TRAIN:].astype(int)


		# naive weather prediction: tomorrow has the same weather as today
		deltas = np.zeros((len(test)-1, data.shape[1]))
		for i in range(0, len(test)-1):
			deltas[i] = np.abs(data[NUM_TRAIN + i - 1] - data[NUM_TRAIN + i])

		mean_delta = np.mean(deltas, axis=0)
		score = np.sqrt(np.sum(np.power(mean_delta / std, 2.0)))
		print("no time naive: " + str(round(score, 3)))
		print(mean_delta)

		for ii in HMM_CLASSES:
			# Run Gaussian HMM
			print("fitting to HMM, c=" + str(ii))

			# Make an HMM instance and execute fit
			hmm_model = hmm.GaussianHMM(n_components=ii, n_iter=10000).fit(train)
			hmm_chain = hmm_model.predict(train)
			print(hmm_model.transmat_)
			print(hmm_model.means_)
			

			# calculate year "centers"
			centers = np.zeros((ii,5))
			for i in range(0, ii):
				select = hmm_chain[::2] == i
				avg_tx = np.mean(dates[::2,1][select])
				avg_ty = np.mean(dates[::2,2][select])

				avg_dates = np.arctan2(avg_ty, avg_tx) / (np.pi * 2.0)
				if avg_dates < 0:
					avg_dates = 1.0 + avg_dates
				avg_temp = np.mean(orig_data[::2,1][select])
				avg_humi = np.mean(orig_data[::2,7][select])
				avg_pres = np.mean(orig_data[::2,10][select])
				avg_rain = np.mean(np.power(orig_data[::2,17][select], 0.333))
				centers[i,0] = avg_dates
				centers[i,1] = avg_temp
				centers[i,2] = avg_humi
				centers[i,3] = avg_pres
				centers[i,4] = avg_rain

			# print orig_data vs time, rel: classes
			plot.plot2d_hmm((dates[::2,0], orig_data[::2,1]), 
						    hmm_chain[::2], centers[:,[0,1]],
						    DATASET+"-timeless-temperature" + str(ii), hmm_model, a=0.5)
			plot.plot2d_hmm((dates[::2,0], orig_data[::2,7]), 
							hmm_chain[::2], centers[:,[0,2]],
							DATASET+"-timeless-humidity" + str(ii), hmm_model, a=0.5)
			plot.plot2d_hmm((dates[::2,0], orig_data[::2,10]), 
							hmm_chain[::2], centers[:,[0,3]],
							DATASET+"-timeless-pressure" + str(ii), hmm_model, a=0.5)
			plot.plot2d_hmm((dates[::2,0], np.power(orig_data[::2,17], 0.333)), 
							hmm_chain[::2], centers[:,[0,4]],
							DATASET+"-timeless-cuberootrain" + str(ii), hmm_model, a=0.5)
			
			
			deltas = np.zeros((len(test)-28, data.shape[1]))

			for i in range(0, len(test)-28-1):
				# predict on window of size 28, but only take last result
				last_class = hmm_model.predict(test[i:i+28])[27]
				pred_value = np.dot(hmm_model.transmat_[last_class], hmm_model.means_)
				deltas[i] = np.abs(pred_value - test[i + 28])

			mean_delta = np.mean(deltas, axis=0)
			score = np.sqrt(np.sum(np.power(mean_delta / std, 2.0)))
			print("HMM, c=" + str(round(ii, 3)) + ": " + str(round(score, 3)))
			print(mean_delta)
			print

