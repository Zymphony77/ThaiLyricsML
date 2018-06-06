# -- coding: utf-8 --

# Use dynamic programming to tokenize the words in the song with preference
# of trying to get longest word (from the front) first and then export as
# a list of words separated by an 'Enter' in each song

import sys

# sys.stdin = open("/Users/oakchawit/Documents/Computer/TestingArea/test.in", "r")
sys.stdout = open("/Users/oakchawit/Documents/Computer/TestingArea/test.out", "w")

# Fixing misspelling words
fix = {'รึ': 'หรือ', 'มั้ย': 'ไหม', 'ปาฏิหารย์': 'ปาฏิหาริย์', 'น่ะ': 'นั้น',
		'เพึ่ง': 'เพิ่ง', 'ซะ': 'เสีย', 'เนี้ย': 'นี้', 
		'ทรมาณ': 'ทรมาน', 'ใว้': 'ไว้', 'ใหน': 'ไหน', 'คุน': 'คุณ', 
		'จิง': 'จริง', 'ล่ะ': 'เล่า', 'ผุ้': 'ผู้'}

dictionary = set()
maxlen = 0

# Import Thai words list to dictionary	--	'ฃฃ' = space, 'ฅฅ' = enter
for line in open("ThaiWordList (Modified).txt", "r"):
	line = line.strip()
	maxlen = max(maxlen, len(line))
	dictionary.add(line)

for song in open("SongList.txt"):
	song = song.strip();

	lyrics = ''
	word = []
	dp = [0]

	# Import lyrics from the input
	for line in open('Data/' + song + '.txt'):
		line = line.strip() + 'ฅฅ'
		for each in '!#*()-_+[]}{\\:;\'\",./?':
			line = line.replace(each, '')
		line = line.replace(' ', 'ฃฃ')
		line = line.replace('ๆ', 'ๆๆ')
		lyrics += line

	# Dynamic Programming for selecting longest words at the front
	lyrics = lyrics[::-1]
	for i in range(len(lyrics)):
		# if lyrics[i] == 'ๆ' and dp[i] != -1:
		# 	dp.append(i)
		# 	continue

		currentWord = lyrics[i]
		prev = -1
		for j in range(i - 1, max(-1, i - maxlen), -1):
			currentWord = currentWord + lyrics[j]
			if (currentWord in dictionary or currentWord in fix) and dp[j] != -1:
				prev = j
		dp.append(prev)

	tmp = len(lyrics)
	while tmp > 0:
		if tmp == -1:
			break

		if lyrics[dp[tmp]: tmp][::-1] in fix:
			word = [fix[lyrics[dp[tmp]: tmp][::-1]]] + word
		else:
			word = [lyrics[dp[tmp]: tmp:][::-1]] + word
		tmp = dp[tmp]

	# for i in range(len(word)):
	# 	if word[i] == 'ๆ':
	# 		for j in range(i - 1, -1, -1):
	# 			if word[j] != 'ฃฃ' and word[j] != 'ฅฅ':
	# 				word[i] = word[j]
	# 				break
	word = word[::-1]

	print(*word, sep = '\n', end = '\n\n')

	# print(song + ":", end = ' ')
	# if dp[-1] == -1:
	# 	print("NO")
	# else:
	# 	print("YES")

	# print(*word)
