from csv import reader
from csv import writer
import emoji
y2016 = None
y2017 = None
y2018 = None
y2019 = None
y2020 = None
full_list = []
remoovale_letters = list('\'\":/\\[-01234–56”7’89”:”#@%_‘“–)(}{+~=•?&$;-]*')
possible = list('qwertyuiopasdfghjklzxcvbnm#@,.?!QWERTYUIOPASDFGHJKLZXCVBNM ')
num2words1 = {'0': 'zero','1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
numbers = '0123456789'
with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/Tweets_coindesk_2016_2020.csv', 'r',newline='') as obj:
	csv_file = reader(obj)
	indexes = [2,5,6]
	test_val = False
	for row in list(csv_file):
		#print(row[2])

		full_list.append([item for item in row if row.index(item) in indexes])
		temp_value = full_list[-1][0]
		start_ind = temp_value.find('https://')
		end_ind = None
		if start_ind == -1:
			pass
		else:
			while start_ind != -1:
				for index, letter in enumerate(temp_value):
					if index < start_ind:
						pass
					else:
						if letter == ' ':
							end_ind = index
							break
				if end_ind == None:
					end_ind = len(temp_value)
				#print('start end:',start_ind,end_ind)
				temp_value = temp_value[:start_ind]+temp_value[end_ind+1:]
				start_ind = temp_value.find('https://')
				end_ind = None
				#print(temp_value)
		preporcessed_tweet = ''
		for letter in temp_value:
			if letter in possible:
				preporcessed_tweet += letter
			if letter in emoji.UNICODE_EMOJI['en']:
				preporcessed_tweet += ' '
				preporcessed_tweet += letter
				preporcessed_tweet += ' '
			if letter in numbers:
				preporcessed_tweet += ' '
				preporcessed_tweet += num2words1[letter]
				preporcessed_tweet += ' '
		if test_val != False:
			if len(full_list[-1]) == 3:
				full_list[-1][1] = int(full_list[-1][1])
				full_list[-1][2] = int(full_list[-1][2])
			if len(full_list[-1]) == 2:
				full_list[-1][1] = int(full_list[-1][1])
		preporcessed_tweet = preporcessed_tweet.lower()
		preporcessed_tweet = preporcessed_tweet.strip()
		full_list[-1][0] = preporcessed_tweet
		test_val = True
for index, i_elm in enumerate(full_list):
		if len(i_elm) == 2:
			temp_1 = i_elm[0]
			temp_3 = i_elm[1]
			full = [temp_1, 0 ,temp_3]
			full_list[index] = full


sotred_list = sorted(full_list[1:], key=lambda x: x[2])
sotred_list.insert(0,full_list[0])
with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/twitter_2016_2020.csv', 'w', newline='') as obj:
	fill_file = writer(obj)
	fill_file.writerows(sotred_list)

