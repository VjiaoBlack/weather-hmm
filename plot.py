import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import offsetbox
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def rotate(angle, ax):
	ax.view_init(azim=angle)

def plot3d_anim(data, title, color=None):
	cmap = matplotlib.cm.get_cmap('inferno')
	if color is None:
		color = np.zeros((len(data)))
	fig = plt.figure(figsize=(8,5), dpi=125)
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(xs=data[:,0], ys=data[:,1], zs=data[:,2], s=0.5, linewidth=1, cmap=cmap, c=color, alpha=0.5) # color is defined by 1st param
	plt.show()
	plt.title(title)
	plt.tight_layout()
	anim = animation.FuncAnimation(fig, rotate, frames=np.arange(0,360,3),interval=80, fargs=(ax, ))
	anim.save(title+"-anim.gif", dpi=125, writer='imagemagick')
	plt.close()

def plot3d(data, title, color=None):
	cmap = matplotlib.cm.get_cmap('inferno')
	if color is None:
		color = np.zeros((len(data)))
	fig = plt.figure(figsize=(8,5), dpi=125)
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(xs=data[:,0], ys=data[:,1], zs=data[:,2], s=0.5, linewidth=1, cmap=cmap, c=color, alpha=0.5) # color is defined by 1st param
	plt.show()
	plt.title(title)
	plt.tight_layout()
	plt.savefig(title+".png")
	plt.close()

def plot2d(data, title, color=None):
	cmap = matplotlib.cm.get_cmap('inferno')
	if color is None:
		color = np.zeros((len(data)))
		print("NONE")
	plt.figure(figsize=(8,5), dpi=125)
	plt.scatter(data[:,0], data[:,1], s=0.5, linewidth=1, cmap=cmap, c=color, alpha=0.5) # color is defined by 1st param
	plt.show()
	plt.title(title)
	plt.tight_layout()
	plt.savefig(title+".png")
	plt.close()