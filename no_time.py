import csv
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, cluster)
from sklearn.cross_decomposition import CCA
from hmmlearn import hmm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import warnings

import collections

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf, suppress=True, precision=3)

for DATASET in ["LGA", "SFO", "MDW", "ORD"]:
	# Replace the data with the specific dataset you want to examine
	with open("data/" + DATASET + ".csv", 'rb') as MDWcsv:
		# np.set_printoptions(threshold=np.inf, suppress=True)
		MDWdata = csv.reader(MDWcsv, delimiter=',')
		MDWdata = np.asarray(list(MDWdata))
		MDWdata = MDWdata[1:]
		MDWdata = np.delete(MDWdata, [0, 18, 21, 22] ,1)
		MDWdata[MDWdata == 'T'] = 0.01 # T is when there is rain but not enough to be measured
		MDWdata = MDWdata[np.all(MDWdata != '',axis=1)]
		# MDWdata = np.concatenate((np.full((1,4000),0).T, MDWdata.astype(np.float)[0:4000:]),axis=1)
		data = MDWdata.astype(np.float)[:,6:]




		print(data.shape)

		print("Using PCA")
		print("\n\n\n" + DATASET + "#######################")
		print(data.shape)
		print
		pca = decomposition.PCA(n_components=2)

		train = pca.fit_transform(data)

		emb = train[::]#[0:1080:]
		label = data[::]#[0:1080:]

		print(emb.shape)

		cmap = matplotlib.cm.get_cmap('inferno')


		plt.figure(figsize=(8,5), dpi=200)	
		plt.scatter(emb[:,0], emb[:,1], s=0.5, linewidth=1, c=cmap((label[:,1] - np.min(label[:,1])) / (np.max(label[:,1]) - np.min(label[:,1]))), alpha=0.5) # color is defined by 7th param
		plt.show()
		title = DATASET + "humidity-notime.png"
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
			stds = np.zeros((ii, 13)) 
			for i in range(0, ii):
				stds[i] = np.sqrt(np.diag(GHMM.covars_[i]))

			for i in range(0,len(adata)):
				z = (adata[i][0:13].reshape(1,13).repeat(ii,axis=0) - GHMM.means_)
				z = z / stds

				zm = np.zeros(ii)
				for j in range(0,ii):
					zm[j] = np.dot(z[j], z[j])
				
				adata[i][13] = np.argmin(zm)

			train = adata[0:16000,13].astype(int)
			test  = adata[16000:,13].astype(int)
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

				deltas = np.zeros((len(test)-n-1, 13))

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
						deltas[test_i] = np.repeat(-1, 13)

				# print("right: " + str(num_right) + " |wrong: " + str(num_wrong) + "|error: " + str(num_error))
				print("n=" + str(n)    +'\t\t{:.2f}'.format((float(num_right) / float(num_right + num_wrong + num_error))))	
				print(np.mean(deltas[deltas[:,0] > -0.1], axis=0))


			print("naive:")

			# um. I guess then we predict using the ngrams???

			deltas = np.zeros((len(test)-n-1, 13))

			for test_i in range(0, len(test)-n-1):
				deltas[test_i] = np.abs(data[16000 + test_i + n - 1] - data[16000 + test_i + n])

			# print("right: " + str(num_right) + " |wrong: " + str(num_wrong) + "|error: " + str(num_error))
			print(np.mean(deltas[deltas[:,0] > -0.1], axis=0))