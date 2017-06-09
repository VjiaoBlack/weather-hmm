import warnings
import csv
import numpy as np
from sklearn import decomposition
from hmmlearn import hmm

import plot
import utils

DATASETS = ["LGA", "SFO", "MDW", "ORD"]
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
		data = utils.format_data(our_csv)[0]
		orig_data = data

		print("\n#####  " + DATASET + "  #####")
		print(data.shape)
		print

		# overall standard deviations of data
		std = np.std(data, axis=0)

		# HMM class estimates
		train = data[0:NUM_TRAIN].astype(int)
		test  = data[NUM_TRAIN:].astype(int)

		# naive weather prediction: tomorrow has the same weather as today
		deltas = np.zeros((len(test)-1, 21))
		for i in range(0, len(test)-1):
			deltas[i] = np.abs(data[NUM_TRAIN + i - 1] - data[NUM_TRAIN + i])

		mean_delta = np.mean(deltas, axis=0)
		score = np.sqrt(np.sum(np.power(mean_delta / std, 2.0)))
		print("naive: " + str(round(score, 3)))
		print(mean_delta)
