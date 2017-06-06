import warnings
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import offsetbox, cm
from sklearn import decomposition
from hmmlearn import hmm


warnings.filterwarnings("ignore")
for DATASET in ["LGA", "SFO", "ORD", "MDW"]:
	# Replace the data with the specific dataset you want to examine
	with open("data/" + DATASET + ".csv", 'rb') as MDWcsv:
		np.set_printoptions(threshold=np.inf, suppress=True, precision=3)
		MDWdata_orig = csv.reader(MDWcsv, delimiter=',')
		MDWdata_orig = np.asarray(list(MDWdata_orig))
		# If data has "min / max / mean", just keep mean.
		MDWdata_orig = np.delete(MDWdata_orig, [0, 18, 21] ,1)
		MDWdata = MDWdata_orig[1:]
		MDWdata = MDWdata[np.all(MDWdata != '',axis=1)]
		MDWdata[MDWdata == 'T'] = 0.001 # T is when there is rain but not enough to be measured
		data = MDWdata.astype(np.float)

		print(data.shape)
		print(MDWdata_orig[0])
		print(data[0])



		# change degree value to an actually meaningful value... ugh
		data = np.concatenate((data,np.zeros((len(data), 1))), axis=1)
		for i in range(0, len(data)):
			angle = np.radians(float(data[i][19]))
			data[i][19] = np.cos(angle) * float(data[i][16])
			data[i][20] = np.sin(angle) * float(data[i][16])


		# num components thingy
		for ii in [2,3,4,5,6,10]:


			print("Using PCA")

			# Run Gaussian HMM
			print("fitting to HMM and decoding ...")

			# Make an HMM instance and execute fit
			GHMM = hmm.GaussianHMM(n_components=ii, covariance_type="diag", n_iter=10000).fit(data)

			# Predict the optimal sequence of internal hidden state
			hidden_states = GHMM.predict(data)

			print("done")

			# Print trained parameters and plot
			print("Transition matrix")
			print(GHMM.transmat_)
			print

			print("Means and vars of each hidden state")
			for i in range(GHMM.n_components):
			    print("{0}th hidden state".format(i))
			    print("mean = " + str(GHMM.means_[i]))
			    print("std = " + str(np.sqrt(np.diag(GHMM.covars_[i]))))
			    print

			# let's plot the shifted data
			adata = np.concatenate((data,np.zeros((len(data), 1))), axis=1)
			stds = np.zeros((ii, 21)) 
			for i in range(0, ii):
				stds[i] = np.sqrt(np.diag(GHMM.covars_[i]))

			for i in range(0,len(adata)):
				z = (adata[i][0:21].reshape(1,21).repeat(ii,axis=0) - GHMM.means_)
				z = z / stds

				zm = np.zeros(ii)
				for j in range(0,ii):
					zm[j] = np.dot(z[j], z[j])
				
				adata[i][21] = np.argmin(zm)




			pca = decomposition.PCA(n_components=2)

			train = pca.fit_transform(adata[:,:-3])

			emb = train
			label = adata

			print(adata.shape)

			cmap = matplotlib.cm.get_cmap('inferno')


			plt.figure(figsize=(8,5), dpi=200)	
			for i in range(0, ii):
				select = adata[:,21] == i
				plt.plot(np.mean(emb[select][:,0]), np.mean(emb[select][:,1]), marker='.', linewidth=1, markersize=10, c=cmap(float(i)/float(ii-1)), markeredgecolor="#5555ff")
				plt.annotate(str(i), (np.mean(emb[select][:,0]), np.mean(emb[select][:,1])), color="#55ff55")
				plt.scatter(emb[select][:,0], emb[select][:,1], s=0.5, linewidth=1,  c=cmap(float(i)/float(ii-1)), alpha=0.5) 
			
			plt.show()
			title = DATASET + "-hmm-classes-" + str(ii) + ".png"
			if title is not None:
				 plt.title(title)
			plt.savefig(title)
			plt.close()


