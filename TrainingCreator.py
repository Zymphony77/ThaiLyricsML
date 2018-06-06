# Take a list of words and output in a form of 7 previous word + 1 result

import sys

sys.stdin = open("/Users/oakchawit/Documents/Computer/TestingArea/test.out", "r")
sys.stdout = open("Data.txt", "w")

dictionary = dict()
previous = list()
cnt = 0

# Importing dictionary
for line in open("ThaiWordList (Modified).txt", "r"):
	dictionary[line.strip()] = cnt
	cnt += 1

# Change format
for line in sys.stdin:
	line = line.strip()

	if len(line) == 0:
		previous = list()
		continue

	if len(previous) == 7:
		print(*previous, end = ' ')
		print(dictionary[line])
		previous = previous[1:]
	previous.append(dictionary[line])