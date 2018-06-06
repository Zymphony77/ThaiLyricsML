# Used for measuring Accuracy of the model

import sys
import math
import numpy as np

# sys.stdin = open("/Users/oakchawit/Documents/Computer/TestingArea/test.in", "r")
# sys.stdout = open("/Users/oakchawit/Documents/Computer/TestingArea/test.out", "w")

# Sigmoid function
def sigmoid(x):
	return 1 / (1 + math.e ** (-x))

# Cost and Prediction
def nnCostFunction(X, y, theta1, theta2):
	m = X.shape[0]

	# Initialize
	print('-> Start')
	cost = 0
	theta1grad = np.zeros(theta1.shape)
	theta2grad = np.zeros(theta2.shape)

	# Forward propagation
	print('-> Forward Propagation (X -> z2)...')
	ones = np.ones((X.shape[0], 1))
	z2 = np.matmul(np.concatenate((ones, X), axis = 1), theta1.T)
	a2 = sigmoid(z2)
	print('-> Forward Propagation (a2 -> z3)...')
	ones = np.ones((a2.shape[0], 1))
	z3 = np.matmul(np.concatenate((ones, a2), axis = 1), theta2.T)
	a3 = sigmoid(z3)

	# Calculate cost function WITHOUT regularization
	print('-> Calculating Cost:', end = ' ')
	cost = sum(sum(- y * np.log(a3) - (1 - y) * np.log(1 - a3))) / m
	print(cost)

	return (a3, cost)


print("Initiate...\n")

# Import Dictionary
print("Importing dictionary...")

n = 0						# Number of words <features>
wtnDictionary = dict()		# Word-to-number
ntwDictionary = dict()		# Number-to-word

for line in open("ThaiWordList (Modified).txt", "r"):
	line = line.strip()
	wtnDictionary[line] = n
	ntwDictionary[n] = line
	n += 1

print("Dictionary imported\n")

# Get testing set
print("Importing testing set...")

m = 0						# Number of testing data
tmp = 0
for line in open("Testing.txt", "r"):
	if line.strip() != '':
		m += 1

testing = np.zeros((m, n))
result = np.zeros((m, n))

# Import PCA Parameter
print("Importing PCA Parameter...\n")

n = 0
k = 0
for line in open("PCA Parameter.txt"):
	if n == 0:
		k = len(line.strip().split())
	n += 1

Ureduce = np.zeros((n, k))
n = 0
for line in open("PCA Parameter.txt"):
	Ureduce[n, :] = np.array([float(e) for e in line.strip().split()])
	n += 1

# Import the model
print("Import theta...")
NN = open("NNparameter.txt")

x, y = [int(e) for e in NN.readline().strip().split()]
theta1 = np.zeros((x, y))
for i in range(x):
	theta1[i, :] = np.array([float(e) for e in NN.readline().strip().split()])
print("Theta1 initialized:", x, y)

x, y = [int(e) for e in NN.readline().strip().split()]
theta2 = np.zeros((x, y))
for i in range(x):
	theta2[i, :] = np.array([float(e) for e in NN.readline().strip().split()])
print("Theta2 initialized:", x, y, "\n")

for line in open("Testing.txt", "r"):
	data = [round(float(e)) for e in line.strip().split()]

	weight = 1
	for each in data[:-1]:
		testing[tmp, each] = weight ** 2
		weight += 1
	result[tmp, data[-1]] = 1

	tmp += 1

print("Testing set imported\n")

# Import mean of training data and perform mean normalization
mean = np.array([float(e) for e in \
	open("TrainingMean.txt").readline().strip().split()])
testing = testing - mean

# Perform PCA
print("Reducing Dimension:", testing.shape[1], '->', Ureduce.shape[1])
# print("(99.99% of variance is retained)\n")
testing = np.matmul(testing, Ureduce)

print("Number of testing samples:", m)
print("Number of input features:", k)
print("Number of neurons of hidden layer:", k)
print("Number of outputs:", n, '\n')

print("Evaluating the result...")

# Testing
prediction, cost = nnCostFunction(testing, result, theta1, theta2)

result = np.argmax(result, axis = 1)
prediction = np.argmax(prediction, axis = 1)
accuracy = np.sum(prediction == result) / result.shape[0]

print("Cost:", cost)
print("Accuracy:", accuracy)

# Show [actual result] vs [prediction] with context
data = open("Testing.txt")

sys.stdout = open("/Users/oakchawit/Documents/Computer/TestingArea/test.out", "w")
for i in range(prediction.shape[0]):
	print(*[ntwDictionary[float(e)] for e in data.readline().strip().split()[:-1]], sep = '', end = '\t|\t')
	print(ntwDictionary[result[i]], \
		ntwDictionary[prediction[i]], sep = '\t')
