import csv
import numpy as np
import utils
import plot

DATASETS = ["LGA", "SFO", "MDW", "ORD"]

for DATASET in DATASETS:
	with open("data/" + DATASET + ".csv", 'rb') as rawcsv:

		# np.set_printoptions(threshold=np.inf, suppress=True)
		data_orig = csv.reader(rawcsv, delimiter=',')
		data, dates, wind = utils.format_data(data_orig)

		date_vector = dates[:,0]

		date_vector *= 12.0
		date_vector += 1

		print(data.shape)
		print(date_vector.shape)

		plot.plot2d((date_vector, data[:,1]), DATASET + "-temp-time", c="black", a=0.2)
		plot.plot2d((date_vector, data[:,7]), DATASET + "-humidity-time", c="black", a=0.2)
		plot.plot2d((date_vector, wind), DATASET + "-winddir-time", c="black", a=0.2)
		plot.plot2d((date_vector, wind), DATASET + "-winddir-speed", c=data[:,16], a=0.2)
		plot.plot2d((date_vector, wind), DATASET + "-winddir-moisture-time", c=data[:,7], a=0.2)

