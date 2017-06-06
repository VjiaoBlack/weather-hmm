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

warnings.filterwarnings("ignore")
for DATASET in ["LGA", "SFO", "ORD"]:
	# Replace the data with the specific dataset you want to examine
	with open("data/" + DATASET + ".csv", 'rb') as MDWcsv:
		np.set_printoptions(threshold=np.inf, suppress=True, precision=3)
		MDWdata_orig = csv.reader(MDWcsv, delimiter=',')
		MDWdata_orig = np.asarray(list(MDWdata_orig))
		# If data has "min / max / mean", just keep mean.
		MDWdata_ = np.delete(MDWdata_orig, [1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 21], 1)
		MDWdata_[MDWdata_ == 'T'] = 0.001 # T is when there is rain but not enough to be measured
		MDWdata_ = MDWdata_[np.all(MDWdata_ != '',axis=1)]

		print(MDWdata_[0])
		# 'CST'
		# 'Mean TemperatureF'				0
		# 'MeanDew PointF'
		# 'Mean Humidity'					2
 		# 'Mean Sea Level PressureIn'
		# 'Mean VisibilityMiles'
		# 'Mean Wind SpeedMPH'
		# 'Max Gust SpeedMPH'
		# 'PrecipitationIn'					7
		# 'CloudCover'
		# 'WindDirDegrees'
		# 
		# Predict mean temp, precipitation, and mean humidity.


		MDWdata_ = MDWdata_[1:]
		MDWdata = np.delete(MDWdata_, [0], 1)



		MDWdata_ = MDWdata_.astype(np.float)
		data = MDWdata.astype(np.float)

		print("\n\n\n" + DATASET + "#######################")
		print(data.shape)
		print

		for ii in [2,4,6,10]:
			# Run Gaussian HMM
			print("fitting to HMM and decoding ... classes=" + str(ii))

			# Make an HMM instance and execute fit
			GHMM = hmm.GaussianHMM(n_components=ii, covariance_type="diag", n_iter=10000).fit(data)

			# Predict the optimal sequence of internal hidden state
			hidden_states = GHMM.predict(data)

			# print("done")

			# Print trained parameters and plot
			print("Transition matrix")
			print(GHMM.transmat_)
			print

			# Sets the 10th position equal to the hmm class
			adata = np.concatenate((data,np.zeros((len(data), 1))), axis=1)
			stds = np.zeros((ii, 10)) 
			for i in range(0, ii):
				stds[i] = np.sqrt(np.diag(GHMM.covars_[i]))

			for i in range(0,len(adata)):
				z = (adata[i][0:10].reshape(1,10).repeat(ii,axis=0) - GHMM.means_)
				z = z / stds

				zm = np.zeros(ii)
				for j in range(0,ii):
					zm[j] = np.dot(z[j], z[j])
				
				adata[i][10] = np.argmin(zm)

			train = adata[0:8000,10].astype(int)
			test  = adata[8000:,10].astype(int)
			ngram = collections.defaultdict(dict)

			for n in [2,3,4,7,14]:
				for i in range(0, len(train)-n):
					key = tuple(train[i:i+n])
					if train[i+1] in ngram[key]:
						ngram[key][train[i+1]] += 1
					else:
						ngram[key][train[i+1]] = 1

				# um. I guess then we predict using the ngrams???
				num_right = 0 # right
				num_wrong = 0 # wrong
				num_error = 0 # does not exist	

				deltas = np.zeros((len(test)-n-1, 10))

				for test_i in range(0, len(test)-n-1):
					key = tuple(test[test_i:test_i+n])

					if key in ngram:
						pred = max(ngram[key], key=ngram[key].get)

						if pred == test[test_i+n]:
							num_right += 1
						else:
							num_wrong += 1

						deltas[test_i] = np.abs(GHMM.means_[pred] - data[8000 + test_i + n])

					else:
						num_error += 1
						deltas[test_i] = np.repeat(-1, 10)

				# print("right: " + str(num_right) + " |wrong: " + str(num_wrong) + "|error: " + str(num_error))
				print("n=" + str(n)    +'\t\t{:.2f}'.format((float(num_right) / float(num_right + num_wrong + num_error))))	
				print(np.mean(deltas[deltas[:,0] > -0.1], axis=0))


			print("naive:")

			# um. I guess then we predict using the ngrams???

			deltas = np.zeros((len(test)-n-1, 10))

			for test_i in range(0, len(test)-n-1):
				deltas[test_i] = np.abs(data[8000 + test_i + n - 1] - data[8000 + test_i + n])

			# print("right: " + str(num_right) + " |wrong: " + str(num_wrong) + "|error: " + str(num_error))
			print(np.mean(deltas[deltas[:,0] > -0.1], axis=0))