import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import offsetbox
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from hmmlearn import hmm

def rotate(angle, ax):
	ax.view_init(azim=angle)

def plot3d_anim(data, title, c=None, a=1):
	cmap = matplotlib.cm.get_cmap('inferno')
	if c is None:
		c = np.zeros((len(data)))
	fig = plt.figure(figsize=(8,5), dpi=125)
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(xs=data[0], ys=data[1], zs=data[2], s=0.5, linewidth=1, cmap=cmap, c=c, alpha=a) # c is defined by 1st param
	plt.show()
	plt.title(title)
	plt.tight_layout()
	anim = animation.FuncAnimation(fig, rotate, frames=np.arange(0,360,3),interval=80, fargs=(ax, ))
	anim.save(title+"-anim.gif", dpi=125, writer='imagemagick')
	plt.close()

def plot3d(data, title, c=None, a=1):
	cmap = matplotlib.cm.get_cmap('inferno')
	if c is None:
		c = np.zeros((len(data)))
	fig = plt.figure(figsize=(8,5), dpi=125)
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(xs=data[0], ys=data[1], zs=data[2], s=0.5, linewidth=1, cmap=cmap, c=c, alpha=a) # c is defined by 1st param
	plt.show()
	plt.title(title)
	plt.tight_layout()
	plt.savefig(title+".png")
	plt.close()

def plot2d(data, title, c=None, a=1):
	cmap = matplotlib.cm.get_cmap('inferno')
	if c is None:
		c = np.zeros((len(data)))
	plt.figure(figsize=(8,5), dpi=125)
	plt.scatter(data[0], data[1], s=0.5, linewidth=1, cmap=cmap, c=c, alpha=a) # c is defined by 1st param
	plt.show()
	plt.title(title)
	plt.tight_layout()
	plt.savefig(title+".png")
	plt.close()

def plot2d_hmm(data, labels, centers, title, model, a=1):
	num_classes = len(model.means_)
	cmap = matplotlib.cm.get_cmap('inferno')

	plt.figure(figsize=(8,5), dpi=125)
	for i in range(0, num_classes):
		select = labels == i
		plt.scatter(data[0][select], data[1][select], s=0.5, linewidth=1, c=cmap(float(i)/float(num_classes)), alpha=a)
		plt.plot(centers[i,0], centers[i,1], marker='.', markersize=15, markeredgecolor=None, linewidth=2, c=cmap(float(i)/float(num_classes)), alpha=1)
		plt.annotate(str(i), xy=(centers[i,0], centers[i,1]), color="black", size=15)

	plt.show()
	plt.tight_layout()
	plt.savefig(title+".png")
	plt.close()

def plot3d_hmm(data, labels, title, model, a=1):
	num_classes = len(model.means_)
	cmap = matplotlib.cm.get_cmap('inferno')

	fig = plt.figure(figsize=(8,5), dpi=125)
	ax = fig.add_subplot(111,projection='3d')
	for i in range(0, num_classes):
		select = labels == i
		ax.scatter(xs=data[0][select], ys=data[1][select], zs=data[2][select], s=0.5, linewidth=1, c=cmap(float(i)/float(num_classes)), alpha=a)

	plt.show()
	plt.tight_layout()
	plt.savefig(title+".png")
	plt.close()

# so slow
def plot3d_hmm_anim(data, labels, title, model, a=1):
	num_classes = len(model.means_)
	cmap = matplotlib.cm.get_cmap('inferno')

	fig = plt.figure(figsize=(8,5), dpi=125)
	ax = fig.add_subplot(111,projection='3d')
	for i in range(0, num_classes):
		select = labels == i
		ax.scatter(xs=data[0][select], ys=data[1][select], zs=data[2][select], s=0.5, linewidth=1, c=cmap(float(i)/float(num_classes)), alpha=a)

	plt.show()
	plt.tight_layout()
	anim = animation.FuncAnimation(fig, rotate, frames=np.arange(0,360,3),interval=80, fargs=(ax, ))
	anim.save(title+"-anim.gif", dpi=125, writer='imagemagick')
	plt.close()