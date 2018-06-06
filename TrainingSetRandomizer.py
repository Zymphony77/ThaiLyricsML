# Divide data to Training, Testing, and Cross-validation set

import sys
import random

cnt = 0
btrain = []
bcv = set()
btest = set()

for line in open("Data.txt", "r"):
	cnt += 1

for i in range(int(cnt * 0.2)):
	while True:
		tmp = random.randint(0, cnt - 1)
		if not(tmp in btest):
			btest.add(tmp)
			break

for i in range(int(cnt * 0.2)):
	while True:
		tmp = random.randint(0, cnt - 1)
		if not(tmp in btest) and not(tmp in bcv):
			bcv.add(tmp)
			break

cnt = 0

training = open("Training.txt", "w")
test = open("Testing.txt", "w")
cv = open("CrossValidation.txt", "w")

for line in open("Data.txt", "r"):
	if cnt in btest:
		test.write(line)
	elif cnt in bcv:
		cv.write(line)
	else:
		btrain.append(line)
	cnt += 1

random.shuffle(btrain)		# Training set is in random order
for line in btrain:
	training.write(line)