import csv
import random


class PrepareData():

	def __init__(self):
	# Constructor class
		pass

	def read_data(self, filename):
	# Read the file : expect a csv file
		self.data=[]		
		with open(filename,"r") as f:
			reader = csv.reader(f)
			for row in reader:
				self.data.append(row)
			headers = self.data[0]
			self.data = self.data[1:]

		for row in self.data:
			for i in range(1,len(row)):
				row[i]=float(row[i])
		return self.data

	def split_train_test(self, data, split_ratio=0.66):
	# Split data randomly with split_ratio, to train and test data
		size = len(data)
		numberTraining = int(split_ratio*size)
		numberTesting = size-numberTraining
		dataPositive=[]
		dataNegative=[]
		
		for row in data:
			if(row[-1]==0.0):
				dataNegative.append(row)
			else:
				dataPositive.append(row)
		
		self.trainingData=[]
		self.testingData=[]
		numDataPositive=len(dataPositive)		
		while(len(dataPositive) >= numDataPositive//2):
			n=random.randint(0,len(dataPositive)-1)
			self.testingData.append(dataPositive[n])
			del dataPositive[n]

		remaining=numberTesting-len(self.testingData)
		for i in range(0,remaining):
			n=random.randint(0,len(dataNegative)-1)
			self.testingData.append(dataNegative[n])
			del dataNegative[n]

		self.trainingData=dataPositive+dataNegative
		return self.split_xy()

	def split_xy(self):
	# Split train and test to their attributes and targets		
		x_train, x_test, y_train, y_test = [], [], [], []
		for row in self.trainingData:
			x_train.append(row[1:len(row)-1])
			y_train.append(row[-1])
		for row in self.testingData:
			x_test.append(row[1:len(row)-1])
			y_test.append(row[-1])
		return x_train, y_train, x_test, y_test
	

if __name__ == "__main__" :

	file = "../data/final_training_data.csv"
	
	dataObj = PrepareData()
	print(dataObj)
	
	data = dataObj.read_data(file)
	print(len(data))
	
	x_train, y_train, x_test, y_test = dataObj.split_train_test(data, split_ratio=0.75)
	print(len(x_train), len(y_train), len(x_test), len(y_test))
	
