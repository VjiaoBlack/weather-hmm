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
for DATASET in ["LGA", "SFO", "MDW", "ORD"]:
	# Replace the data with the specific dataset you want to examine
	with open("data/" + DATASET + ".csv", 'rb') as MDWcsv:


		np.set_printoptions(threshold=np.inf, suppress=True, precision=3)
		MDWdata_orig = csv.reader(MDWcsv, delimiter=',')
		MDWdata_orig = np.asarray(list(MDWdata_orig))
		# If data has "min / max / mean", just keep mean.
		MDWdata_ = np.delete(MDWdata_orig, [0, 18, 21], 1)
		MDWdata_[MDWdata_ == 'T'] = 0.001 # T is when there is rain but not enough to be measured
		MDWdata_ = MDWdata_[np.all(MDWdata_ != '',axis=1)]

		print(MDWdata_[0])

		MDWdata_ = MDWdata_[1:]
		_data = MDWdata_

		# change degree value to an actually meaningful value... ugh
		_data = np.concatenate((_data,np.zeros((len(_data), 1))), axis=1)
		for i in range(0, len(_data)):
			angle = np.radians(float(_data[i][19]))
			_data[i][19] = np.cos(angle) * float(_data[i][16])
			_data[i][20] = np.sin(angle) * float(_data[i][16])

		MDWdata_ = MDWdata_.astype(np.float)
		_data = _data.astype(np.float)

		for i in reversed(range(1, len(_data))):
			_data[i] -= _data[i-1]

		# concat each data point with 7
		
		data = np.zeros((len(_data)-8, 147))
		for i in range(0, len(data)):
			data[i] = np.concatenate(tuple(_data[i:i+7]))



		print("Using PCA")
		print("\n\n\n" + DATASET + "#######################")
		print(data.shape)
		print
		pca = decomposition.PCA(n_components=2)

		train = pca.fit_transform(data)

		emb = train[::]#[0:1080:]
		label = data[-21:]

		print(emb.shape)

		cmap = matplotlib.cm.get_cmap('inferno')


		plt.figure(figsize=(8,5), dpi=200)	
		plt.scatter(emb[:,0], emb[:,1], s=0.5, linewidth=1, c=cmap((label[:,7] - np.min(label[:,7])) / (np.max(label[:,7]) - np.min(label[:,7]))), alpha=0.5) # color is defined by 7th param
		plt.show()
		title = DATASET + "humidity-concat.png"
		if title is not None:
			 plt.title(title)
		plt.savefig(title)
		plt.close()

		for ii in [2,4,6,10]:
			# Run Gaussian HMM
			print("fitting to HMM and decoding ... classes=" + str(ii))

			# Make an HMM instance and execute fit
			GHMM = hmm.GaussianHMM(n_components=ii, covariance_type="diag", n_iter=20000).fit(data)

			# Predict the optimal sequence of internal hidden state
			hidden_states = GHMM.predict(data)

			# print("done")

			# Print trained parameters and plot
			print("Transition matrix")
			print(GHMM.transmat_)
			print

			# Sets the 10th position equal to the hmm class
			adata = np.concatenate((data,np.zeros((len(data), 1))), axis=1)
			stds = np.zeros((ii, 147)) 
			for i in range(0, ii):
				stds[i] = np.sqrt(np.diag(GHMM.covars_[i]))

			for i in range(0,len(adata)):
				z = (adata[i][0:147].reshape(1,147).repeat(ii,axis=0) - GHMM.means_)
				z = z / stds

				zm = np.zeros(ii)
				for j in range(0,ii):
					zm[j] = np.dot(z[j], z[j])
				
				adata[i][147] = np.argmin(zm)

			train = adata[0:16000,147].astype(int)
			test  = adata[16000:,147].astype(int)
			ngram = collections.defaultdict(dict)

			for n in [1,2,3,4,7,14]:
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

				deltas = np.zeros((len(test)-n-1, 147))

				for test_i in range(0, len(test)-n-1):
					key = tuple(test[test_i:test_i+n])

					if key in ngram:
						pred = max(ngram[key], key=ngram[key].get)

						if pred == test[test_i+n]:
							num_right += 1
						else:
							num_wrong += 1

						deltas[test_i] = np.abs(GHMM.means_[pred] - data[16000 + test_i + n])

					else:
						num_error += 1
						deltas[test_i] = np.repeat(-1, 147)

				# print("right: " + str(num_right) + " |wrong: " + str(num_wrong) + "|error: " + str(num_error))
				print("n=" + str(n)    +'\t\t{:.2f}'.format((float(num_right) / float(num_right + num_wrong + num_error))))	
				print(np.mean(deltas[deltas[:,0] > -0.1], axis=0)[-21:])


			print("naive:")

			# um. I guess then we predict using the ngrams???

			deltas = np.zeros((len(test)-n-1, 147))

			for test_i in range(0, len(test)-n-1):
				deltas[test_i] = np.abs(data[16000 + test_i + n - 1] - data[16000 + test_i + n])

			# print("right: " + str(num_right) + " |wrong: " + str(num_wrong) + "|error: " + str(num_error))
			print(np.mean(deltas[deltas[:,0] > -0.1], axis=0)[-21:])