import numpy
from data_gen import gen
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def visualize(x,y):
	
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x[:,0],x[:,1],y)
	
	plt.show()

def sigmoid(x):
	return	1./(1+numpy.exp(-x))

if __name__=="__main__":
	print("Welcome to gradient descent based logistic regression implementation in python using numpy")
	trainN=5000
	testN=1000
	
	trainx=numpy.random.uniform(0,1,[trainN,2])
	testx=numpy.random.uniform(0,1,[testN,2])
	
	
	trainy=gen(trainx)
	testy=gen(testx)


	#logistic regression part
	a=numpy.array([0.001,0.001])
	bias=0.001
	
	for epoch in range(10000+1):
		#calculating the output
		temp=numpy.dot(trainx,a.T)+bias
		#print(temp.shape)
		output=sigmoid(numpy.dot(trainx,a.T)+bias)
		output_test=sigmoid(numpy.dot(testx,a.T)+bias)
		#loss in terms of empirical risk
		distance=trainy.squeeze()-output
		#print(output.shape)
		#print('shape of distance',distance.shape)
		loss=numpy.mean((distance)**2)
		
		#calculating the updates
		
		coefficient=distance*(output)*(1-output)
		#print(coefficient.shape)
		da1=numpy.mean(coefficient*trainx[:,0])
		da2=numpy.mean(coefficient*trainx[:,1])
		db= numpy.mean(coefficient)
	
		#print(da1,da2,db)
		
		eta=0.05
		
		a[0]=a[0]+eta*da1
		a[1]=a[1]+eta*da2
		bias=bias+eta*db

		if((epoch%5000)==0):
			pass
#			visualize(testx,output_test)
		
		prediction=(output_test>0.5).astype('int32')
		accuracy=numpy.sum((prediction-testy.squeeze()==0).astype('int32'))
		print(loss,accuracy/float(testN)*100)
		time.sleep(1)	
