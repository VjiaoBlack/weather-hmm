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
NUM_CONCAT = 7

np.set_printoptions(formatter={'float': '{:05.2f}'.format})


warnings.filterwarnings("ignore")
for DATASET in DATASETS:
	with open("data/" + DATASET + ".csv", 'rb') as rawcsv:

		our_csv = csv.reader(rawcsv, delimiter=',')
		_data, dates, wind = utils.format_data(our_csv)
		orig_data = np.array(_data)


		print("\n#####  " + DATASET + "  #####")
		print(_data.shape)
		print

		# overall standard deviations of _data
		std = np.std(_data, axis=0)

		data = np.zeros((len(_data)-8, NUM_CONCAT * _data.shape[1]))
		for i in range(0, len(data)):
			data[i] = np.concatenate(tuple(_data[i:i+NUM_CONCAT]))

		# HMM class estimates
		train = data[0:NUM_TRAIN].astype(int)
		test  = data[NUM_TRAIN:].astype(int)

		for ii in HMM_CLASSES:
			# Run Gaussian HMM
			print("fitting to HMM, c=" + str(ii))

			# Make an HMM instance and execute fit
			hmm_model = hmm.GaussianHMM(n_components=ii, n_iter=10000).fit(train)
			hmm_chain = hmm_model.predict(train)
			print(hmm_model.transmat_)
			for i in range(0, ii):	
				print(hmm_model.means_[i][-21:] - hmm_model.means_[i][0:21])
			

			history = 28
			deltas = np.zeros((len(test)-history, 21))

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
						    hmm_chain[::2], np.delete(centers, 2, 1),
						    DATASET + "-concat-temperature" + str(ii), hmm_model, a=0.5)
			plot.plot2d_hmm((dates[::2,0], orig_data[::2,7]), 
							hmm_chain[::2], np.delete(centers, 1, 1),
							DATASET + "-concat-humidity" + str(ii), hmm_model, a=0.5)
			plot.plot2d_hmm((dates[::2,0], orig_data[::2,10]), 
							hmm_chain[::2], centers[:,[0,3]],
							DATASET+"-concat-pressure" + str(ii), hmm_model, a=0.5)
			plot.plot2d_hmm((dates[::2,0], np.power(orig_data[::2,17], 0.333)), 
							hmm_chain[::2], centers[:,[0,4]],
							DATASET+"-concat-cuberootrain" + str(ii), hmm_model, a=0.5)
			
			for i in range(0, len(test)-history-1):
				# predict on window of size history, but only take last result
				last_class = hmm_model.predict(test[i:i+history])[-1]
				pred_value = np.dot(hmm_model.transmat_[last_class], hmm_model.means_)
				deltas[i] = np.abs(pred_value - test[i+history])[-21:]

			mean_delta = np.mean(deltas, axis=0)
			score = np.sqrt(np.sum(np.power(mean_delta / std, 2.0)))
			print("HMM, c=" + str(round(ii, 3)) + ": " + str(round(score, 3)))
			print(mean_delta)
			print

