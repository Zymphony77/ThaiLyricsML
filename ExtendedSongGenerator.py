# -- coding: utf-8 --

# Use NN Parameters as a model to extend the song

import sys
import math
import numpy as np

# sys.stdin = open("/Users/oakchawit/Documents/Computer/TestingArea/test.in", "r")
# sys.stdout = open("/Users/oakchawit/Documents/Computer/TestingArea/test.out", "w")

# Output file
output = open("/Users/oakchawit/Documents/Computer/TestingArea/test.out", "w")

# Sigmoid function
def sigmoid(x):
	return 1 / (1 + math.e ** (-x))

# Next word from previous words
def predict(X, pca, theta1, theta2):
	# Initialize
	# print('-> Start')

	# Reducing dimension
	X = np.matmul(X, pca)

	# Forward propagation
	# print('-> Forward Propagation (X -> z2)...')
	ones = np.ones((X.shape[0], 1))
	z2 = np.matmul(np.concatenate((ones, X), axis = 1), theta1.T)
	a2 = sigmoid(z2)
	# print('-> Forward Propagation (a2 -> z3)...')
	ones = np.ones((a2.shape[0], 1))
	z3 = np.matmul(np.concatenate((ones, a2), axis = 1), theta2.T)
	a3 = sigmoid(z3)

	# Maximum probability
	return (np.max(a3), np.argmax(a3, axis = 1)[0])


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

# Import NN Parameters
print("Importing theta...")
NN = open("NNparameter.txt")

x, y = [int(e) for e in NN.readline().strip().split()]
theta1 = np.zeros((x, y))
for i in range(x):
	theta1[i, :] = np.array([float(e) for e in NN.readline().strip().split()])
print("Theta1 initialized")

x, y = [int(e) for e in NN.readline().strip().split()]
theta2 = np.zeros((x, y))
for i in range(x):
	theta2[i, :] = np.array([float(e) for e in NN.readline().strip().split()])
print("Theta2 initialized\n")

# Import initial lyrics
print("Importing lyrics...")

previous = []
for line in open("/Users/oakchawit/Documents/Computer/TestingArea/test2.in", "r"):
	line = line.strip()

	if len(previous) == 7:
		previous = previous[1:]
	previous.append(wtnDictionary[line])

	# Decode some words as encoded beforehand
	if line == 'ฃฃ':
		output.write(' ')
	elif line == 'ฅฅ':
		output.write('\n')
	elif line == 'ๆๆ':
		output.write('ๆ')
	else:
		output.write(line)

print("Lyrics imported\n")

# Import Mean of Training Data
mean = np.array([float(e) for e in \
	open("TrainingMean.txt").readline().strip().split()])

# Extend the song
print("Generating Song...")

cntWord = 0
minWord = 80
maxWord = 100

while cntWord < maxWord:
	before = np.zeros((1, n))

	weight = 1
	for each in previous:
		before[0, each] = weight * weight
		weight += 1
	before = before - mean		# Mean Normalization

	prob, nxt = predict(before, Ureduce, theta1, theta2)

	print(nxt, prob, end = '\n')

	cntWord += 1

	# Allow to stop early if 'Enter' is predicted
	if ntwDictionary[nxt] == 'ฃฃ':
		output.write(' ')
	elif ntwDictionary[nxt] == 'ฅฅ':
		output.write('\n')
		if cntWord >= minWord:
			break
	elif ntwDictionary[nxt] == 'ๆๆ':
		output.write('ๆ')
	else:
		output.write(ntwDictionary[nxt])

	previous = previous[1:]
	previous.append(nxt)

print("Generating Done!!")

# sys.stdout = open("test.out", "w")
# for i in range(prediction.shape[0]):
# 	print(testing[i, 0], prediction[i, 0], sep = '\t')
