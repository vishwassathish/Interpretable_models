### Chemistry dataset : Analysis ###

# I found the dataset to be very large for analysis through simple lookup.
# Hence, this small script, throwing out some basic details and structure of the given data.

# Attributes
f = open('data.csv', 'r')
attributes = f.readline().strip().split(',')
# 3805 attributes, 1st is Name, last is the target.
print("Number of Attributes : ", attributes.__len__(), "\n", attributes)

# Sample row
row1 = f.readline().strip().split(',')
print("Sample Name : ", row1[0])
print("Sample x_train element : ", row1[1:len(row1)-1])
print("Sample y_train element : ", row1[-1])

# Target analysis
f2 = open('training_data_label.txt', 'r')
Y = []
unknown = 0
for line in f2.readlines()[1:]:
	line = line.strip().split('\t')
	try:
		Y.append(line[1])
	except:
		# 5 such rows
		print("Row with no target : ", line)
		unknown += 1

count = 0
for y in Y:
	if y == '1.0':
		count += 1


print("Total Y : ", len(Y)+unknown) # 9362 total
print("Positive Y : ", count)		# 380 (Very less compared to -ve class)
print("Negative Y : ", len(Y) - count)	# 8977 
print("No Target value given (Unknown) : ", unknown) # 5

f.close()
f2.close()


## General observations ##

'''
-> Huge number of attributes. A considerable amount of time must be spent on figuring out how these attributes might be correlated. Removal of redundant attributes, if any, might be very helpful.

-> Though the number of attributes are huge, we know the exact number and values for each. Hence a complex non linear model like neural network involving unknown number of hidden attributes might not be necessary, unless, simpler ones are not enough.

-> Moreover, the imbalance in the number of positive and negative class data might pose a problem for learning a model. We can, use only as many negative samples as there are positive (probably + or - 100) without inducing a bias in the model. 

-> Since very less proportion of data belong to the positive class, a strong trend from a set of attributes, if observed can considerably solve the
problem of classification with less complex models.

-> Domain knowledge on which attribute is more important or which attributes are probably highly correlated, might save a lot of time.

'''