import warnings
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import offsetbox, cm
from sklearn import decomposition
from sklearn.feature_extraction.text import CountVectorizer
from hmmlearn import hmm
from calendar import monthrange, isleap
import datetime as dt

import collections

import plot
import utils

DATASETS = ["LGA", "SFO", "MDW", "ORD"]
HMM_CLASSES = [2, 4, 6, 8, 10]
NUM_TEST = 20000

np.set_printoptions(threshold=np.inf, suppress=True, precision=2)


warnings.filterwarnings("ignore")
for DATASET in DATASETS:
	with open("data/" + DATASET + ".csv", 'rb') as rawcsv:

		our_csv = csv.reader(rawcsv, delimiter=',')
		data = utils.format_data(our_csv)[0]

		print("\n#####  " + DATASET + "  #####")
		print(data.shape)
		print

		# overall standard deviations of data
		std = np.std(data, axis=0)

		# HMM class estimates
		train = data[0:NUM_TEST].astype(int)
		test  = data[NUM_TEST:].astype(int)

		# naive weather prediction: tomorrow has the same weather as today
		deltas = np.zeros((len(test)-1, 21))
		for i in range(0, len(test)-1):
			deltas[i] = np.abs(data[NUM_TEST + i - 1] - data[NUM_TEST + i])

		mean_delta = np.mean(deltas, axis=0)
		score = np.sqrt(np.sum(np.power(mean_delta / std, 2.0)))
		print("naive: " + str(round(score, 3)))
		print(mean_delta)


		for ii in HMM_CLASSES:
			# Run Gaussian HMM
			print("fitting to HMM, c=" + str(ii))

			# Make an HMM instance and execute fit
			hmm_model = hmm.GaussianHMM(n_components=ii).fit(train)

			deltas = np.zeros((len(test)-28, 21))

			for i in range(0, len(test)-28-1):
				# predict on window of size 28, but only take last result
				pred_class = hmm_model.predict(test[i:i+28])[27]
				pred_value = hmm_model.means_[pred_class]
				deltas[i] = np.abs(pred_value - test[i + 28])

			mean_delta = np.mean(deltas, axis=0)
			score = np.sqrt(np.sum(np.power(mean_delta / std, 2.0)))
			print("HMM, c=" + str(round(ii, 3)) + ": " + str(round(score, 3)))
			print(mean_delta)

