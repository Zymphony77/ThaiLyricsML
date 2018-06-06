# Calculating the Mean of Training Data for Mean Normalization

import sys
import numpy as np

# sys.stdin = open("/Users/oakchawit/Documents/Computer/TestingArea/test.in", "r")
# sys.stdout = open("/Users/oakchawit/Documents/Computer/TestingArea/test.out", "w")

# Import dictionary
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

# Get training set
print("Importing training set...")

m = 0						# Number of training data
tmp = 0
for line in open("Training.txt", "r"):
	if line.strip() != '':
		m += 1

training = np.zeros((m, n))
result = np.zeros((m, n))

for line in open("Training.txt", "r"):
	data = [int(e) for e in line.strip().split()]

	weight = 1
	for each in data[:-1]:
		training[tmp, each] = weight ** 2
		weight += 1
	result[tmp, data[-1]] = 1

	tmp += 1

print("Training set imported\n")

# Calculation mean of trianing data
mean = np.mean(training, axis = 0)

sys.stdout = open("TrainingMean.txt", "w")
print(*mean[:])