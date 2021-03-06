# Continue training (or creating if one does not exists) the model
# by performing mini-batch gradient descent as opposed to batch
# gradient descent in all other files

import sys
import math
import numpy as np

# sys.stdin = open("/Users/oakchawit/Documents/Computer/TestingArea/test.in", "r")
# sys.stdout = open("/Users/oakchawit/Documents/Computer/TestingArea/test.out", "w")

# Sigmoid function
def sigmoid(x):
	return 1 / (1 + math.e ** (-x))

# Diffential of sigmoid function
def sigmoidGradient(x):
	return sigmoid(x) * (1 - sigmoid(x))

# Cost and Gradient
def nnCostFunction(X, y, theta1, theta2, regConst):
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

	# Calculate cost function with regularization
	print('-> Calculating Cost:', end = ' ')
	cost = sum(sum(- y * np.log(a3) - (1 - y) * np.log(1 - a3))) / m \
	+ regConst * (sum(sum(theta1[:,1:] ** 2)) + sum(sum(theta2[:,1:] ** 2))) / (2 * m)
	print(cost)

	# Show Training Accuracy
	result = np.argmax(y, axis = 1)
	prediction = np.argmax(a3, axis = 1)
	accuracy = np.sum(prediction == result) / result.shape[0]
	print('-> Training Accuracy:', accuracy)

	# Calculate differences
	delta3 = a3 - y
	delta2 = np.matmul(delta3, theta2)[:,1:] * sigmoidGradient(z2)

	# Calculate gradients
	print('-> Backward Propagation...')
	ones = np.ones((X.shape[0], 1))
	theta1grad = np.matmul(delta2.T, np.concatenate((ones, X), axis = 1)) / m
	ones = np.ones((a2.shape[0], 1))
	theta2grad = np.matmul(delta3.T, np.concatenate((ones, a2), axis = 1)) /  m

	# Regularization
	print('-> Add Regularization Term...')
	theta1grad[:,1:] += np.multiply(regConst / m, theta1[:,1:])
	theta2grad[:,1:] += np.multiply(regConst / m, theta2[:,1:])

	return (cost, theta1grad, theta2grad)


print("Initiate...\n")

# Import Dictionary
print("Importing dictionary...")

n = 0						# Number of words <features>
ntwDictionary = dict()		# Number-to-word

for line in open("ThaiWordList (Modified).txt", "r"):
	line = line.strip()
	ntwDictionary[n] = line
	n += 1

print("Dictionary imported\n")

# Get PCA Parameters
print("Importing PCA Parameters...")

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

print("PCA Parameters imported\n")

print("Dimension Reduction:", Ureduce.shape[0], '->', Ureduce.shape[1])
print("(99.999999% of variance is retained)\n")

# Handling Theta - Import if exists and Create if not exists
try:
	NN = open("NNparameter.txt")

	print("Import theta...")

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

except:
	print("Initializing theta...")
	epsilon = 0.1
	theta1 = 2 *  epsilon * np.random.rand(k, k + 1) - epsilon
	print("Theta1 initialized")
	theta2 = 2 *  epsilon * np.random.rand(n, k + 1) - epsilon
	print("Theta2 initialized\n")

# Importing Mean of Training
mean = np.array([float(e) for e in \
	open("TrainingMean.txt").readline().strip().split()])

# Details
print("Number of input features:", k)
print("Number of neurons of hidden layer:", k)
print("Number of outputs:", n, '\n')

# Training
partition = 17			# Number of mini-batch
cntIteration = 0
maxIteration = 200
alpha = 0.2
regConst = 1

cost = float('inf')
diff = float('inf')

while cntIteration < maxIteration:
	print("Iteration ", cntIteration + 1, ":", sep = '')

	# Get training set - Each mini-batch has a file of its own
	idx = cntIteration % partition
	print("Importing training set #" + str(idx) + '...')

	m = 0						# Number of training data
	tmp = 0
	for line in open("Training" + str(idx) + ".txt", "r"):
		if line.strip() != '':
			m += 1

	training = np.zeros((m, n))
	result = np.zeros((m, n))

	for line in open("Training" + str(idx) + ".txt", "r"):
		data = [int(e) for e in line.strip().split()]

		weight = 1
		for each in data[:-1]:
			training[tmp, each] = weight ** 2
			weight += 1
		result[tmp, data[-1]] = 1

		tmp += 1

	# Mean Normalization & Dimension Reduction
	training = training - mean
	training = np.matmul(training, Ureduce)

	# Neural Network
	newCost, theta1grad, theta2grad = nnCostFunction(training, result, \
	theta1, theta2, regConst)

	theta1 -= np.multiply(alpha, theta1grad)
	theta2 -= np.multiply(alpha, theta2grad)
	
	diff = cost - newCost
	cost = newCost
	cntIteration += 1
	print(">>> Cost:", cost + diff, "->", cost)

print("Neural Network Completed!!\n")

# Export the model
print("Exporting parameters...")

tmp = sys.stdout
sys.stdout = open("NNparameter.txt", "w")

print(theta1.shape[0], theta1.shape[1])
for i in range(theta1.shape[0]):
	print(*theta1[i, :])

print(theta2.shape[0], theta2.shape[1])
for i in range(theta2.shape[0]):
	print(*theta2[i, :])

sys.stdout.close()
sys.stdout = tmp

print("Exporting Parameters done\n")

print("Program Terminated")
