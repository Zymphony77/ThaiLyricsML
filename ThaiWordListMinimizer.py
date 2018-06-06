# coding: UTF-8

# Minimizing list of all Thai words by removing all words that can be
# created by concatenation of other words to reduce dimension

import sys

# sys.stdin = open("/Users/oakchawit/Documents/Computer/TestingArea/test.in", "r")
sys.stdout = open("NewThaiWord [Modified].txt", "w")

dictionary = {''}

def cutting(word, depth):
	for character in '().ๆ':
		if character in word:
			return True

	if depth > 10:
		if not word in dictionary:
			return False

	if depth > 1:
		if word in dictionary:
			return True

	# Try all combinations of length at least 2 up to 11 words concatenated
	for i in range(2, len(word) - 1):
		if word[:i].strip() in dictionary and cutting(word[i:].strip(), depth + 1):
			return True

	return False

for line in open("ThaiWordList (Modified).txt", "r"):
	dictionary.add(line.strip())

for word in open("ThaiWordList (Modified).txt", "r"):
	word = word.strip()

	if not cutting(word, 1):
		print(word)