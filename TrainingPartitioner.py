# Partitioning the data to parts
# Note: Only for the case: partition | number of training data

import sys

# sys.stdin = open("/Users/oakchawit/Documents/Computer/TestingArea/test.in", "r")
# sys.stdout = open("/Users/oakchawit/Documents/Computer/TestingArea/test.out", "w")

partition = 17
cnt = 0

m = 0
for line in open("Training.txt"):
	m += 1

training = open("Training.txt")

for i in range(partition):
	out = open("Training" + str(i) + '.txt', 'w')

	for j in range(m // partition):
		out.write(training.readline())