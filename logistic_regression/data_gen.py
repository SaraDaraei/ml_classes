import numpy
import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gen(x):
	a=2
	b=-0.5
	c=x[:,0]*a+b
	print(c.shape)
	y=(c-x[:,1])>0
	return y[:,None].astype('int32')




if __name__=="__main__":
	print('Don\'t call directly')
	trainN=5000
	testN=1000
	trainx=numpy.random.uniform(0,1,[trainN,2])
	
	trainy=gen(trainx)
	print(trainy.shape)
	
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(trainx[:,0],trainx[:,1],trainy)
	
	plt.show()
