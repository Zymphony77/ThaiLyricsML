# Calculating PCA Paremeters

import sys
import numpy as np

# sys.stdin = open("/Users/oakchawit/Documents/Computer/TestingArea/test.in", "r")
# sys.stdout = open("/Users/oakchawit/Documents/Computer/TestingArea/test.out", "w")

# Import dictionary
n = 0
for line in open("ThaiWordList (Modified).txt", "r"):
	n += 1

# Import training data
m = 0
tmp = 0
for line in open("Training.txt", "r"):
	if line.strip() != '':
		m += 1

training = np.zeros((m, n))

for line in open("Training.txt", "r"):
	data = [int(e) for e in line.strip().split()]

	weight = 1
	for each in data[:-1]:
		training[tmp, each] = weight ** 2
		weight += 1

	tmp += 1

# Mean normalization
training = training - np.mean(training, axis = 0)

# Calculating PCA Parameter
print("Start Finding Covariance Matrix...")
sigma = np.matmul(training.T, training) / m
print('Covariance Matrix Completed...')
[U, S, V] = np.linalg.svd(sigma)
print('Singular Value Decomposition Completed...')

# Calculate reduced dimension
varRetain = 0
dimCnt = 0
diagonalSum = np.sum(S)

for i in range(S.shape[0]):
	varRetain += S[i] / diagonalSum
	dimCnt += 1
	if varRetain >= 0.9999:
		break

print("Dimension:", S.shape[0], '->', dimCnt)

# Export PCA Parameter
sys.stdout = open("PCA Parameter.txt", "w")
for i in range(U.shape[0]):
	print(*U[i, :dimCnt])