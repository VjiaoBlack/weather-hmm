import warnings
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import offsetbox, cm
from sklearn import decomposition
from hmmlearn import hmm
from calendar import monthrange, isleap
import datetime as dt


warnings.filterwarnings("ignore")
for DATASET in ["LGA", "SFO", "ORD", "MDW"]:
	# Replace the data with the specific dataset you want to examine
	with open("data/" + DATASET + ".csv", 'rb') as MDWcsv:
		np.set_printoptions(threshold=np.inf, suppress=True, precision=3)
		MDWdata_orig = csv.reader(MDWcsv, delimiter=',')
		MDWdata_orig = np.asarray(list(MDWdata_orig))
		# If data has "min / max / mean", just keep mean.
		MDWdata_ = np.delete(MDWdata_orig, [18, 21], 1)
		MDWdata_[MDWdata_ == 'T'] = 0.001 # T is when there is rain but not enough to be measured
		MDWdata_ = MDWdata_[np.all(MDWdata_ != '',axis=1)]
		MDWdata_ = MDWdata_[1:]



		MDWdata = np.delete(MDWdata_, [0], 1)




		# change degree value to an actually meaningful value... ugh
		MDWdata = np.concatenate((MDWdata,np.zeros((len(MDWdata), 1))), axis=1)

		for i in range(0, len(MDWdata)):
			angle = np.radians(float(MDWdata[i][19]))
			MDWdata[i][19] = np.cos(angle) * float(MDWdata[i][16])
			MDWdata[i][20] = np.sin(angle) * float(MDWdata[i][16])


		# change every date to a unit vector
		# integral part is year (currently zero, actually)
		# fractional part is day fraction
		date_vector = np.zeros((len(MDWdata_), 2))
		for i in range(0, len(MDWdata_)):
			date = map(int,MDWdata_[i][0].split("-"))
			# MDWdata_[i][0] = float(date_tup[0]) + float()
			days = (dt.date(date[0], date[1], date[2]) - dt.date(date[0],1,1)).days
			
			if isleap(date[0]): # float(date[0]) +
				metric = (days / 366.0) 
			else:
				metric = (days / 365.0)

			MDWdata_[i][0] = metric * 12.0 + 1.0

			date_vector[i][0] = np.cos(metric * 3.1415926535 * 2.0)
			date_vector[i][1] = np.sin(metric * 3.1415926535 * 2.0)

		MDWdata_ = MDWdata_.astype(np.float)
		data = MDWdata.astype(np.float)


		


		print(DATASET)
		print(data.shape)
		print

		categories = {
			1: "temp", 
			7: "humidity", 
			10: "pressure", 
			16: "wind"
		}






		# for ii in [5,6]:
		# for ii in [5, 6]:
		for ii in [2,3,4,5,6,10]:
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

			print("Means and vars of each hidden state")
			for i in range(GHMM.n_components):
			    print("{0}th hidden state".format(i))
			    print("mean = " + str(GHMM.means_[i]))
			    print("std = " + str(np.sqrt(np.diag(GHMM.covars_[i]))))
			    print

			# Sets the 10th position equal to the hmm class
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

			# graph the data by year and... temperature and class color??
			cmap = matplotlib.cm.get_cmap('inferno')

			for j in categories.keys():
				print(categories[j])

				plt.figure(figsize=(8,5), dpi=200)	
				for i in range(0, ii):
					select = adata[:,21] == i
					x_mean = np.mean(date_vector[select][:,0])
					y_mean = np.mean(date_vector[select][:,1])

					theta = np.arctan2(y_mean, x_mean)
					if theta < 0:
						theta = 3.1415926535 * 2 +  theta

					time = theta / (3.1415926535 * 2.0)

					center_date = (dt.datetime(2017, 1, 1) + dt.timedelta(time * 365.25 - 1))
					center_std = int(np.round( np.sqrt(- 2 * np.log(np.sqrt(x_mean * x_mean + y_mean * y_mean))) * 365.25))

					center_temp = np.mean(adata[select][:,j])
					std_temp = np.std(adata[select][:,j])
					print("class " + str(i) + " centers around " + str(center_date.month) + " " + str(center_date.day) + " std:" + str(center_std) + " y:" + str(np.round(center_temp)) + " std:" + str(np.round(std_temp)))
					time = time * 12.0 + 1.0

					yvar_mean = center_temp

					plt.plot(time, yvar_mean, marker='.', linewidth=0.5, markersize=10, c=cmap(float(i)/float(ii-1)), markeredgecolor="#6666FF")
					plt.scatter(MDWdata_[select][:,0], adata[select][:,j], s=0.5, marker='.', linewidth=1, c=cmap(float(i)/float(ii-1)), alpha=.8) 
					plt.annotate(str(i), (time, yvar_mean), color="#66FF66")
					
				print
				plt.show()
				title = DATASET + "-thickhmm-dateclass-" + categories[j] + str(ii) + ".png"
				if title is not None:
					 plt.title(title)
				plt.savefig(title)
				plt.close()




