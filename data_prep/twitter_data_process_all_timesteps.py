from csv import reader
from csv import writer
import emoji
y2016 = None
y2017 = None
y2018 = None
y2019 = None
y2020 = None
full_list = []
remoovale_letters = list('\'\":/\\[-01234–56”7’89”:”%_‘“–)(}{+~=•?&$;-]*')
possible = list('qwertyuiopasdfghjklzxcvbnm#@,.?!QWERTYUIOPASDFGHJKLZXCVBNM ')
num2words1 = {'0': 'zero','1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
numbers = '0123456789'
tweets_norm = []


with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/twitter_2016_2020.csv', 'r',newline='') as obj:
	csv_file = reader(obj)
	indexes = [2,5,6]
	test_val = False
	iter_tweets = iter(list(csv_file))
	curretn_tweet = next(iter_tweets)
	tweets_norm.append(curretn_tweet)
	curretn_tweet = next(iter_tweets)

with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/bitstamp_min_2016_2020.csv', 'r',newline='') as obj:
	csv_file = reader(obj)
	indexes = [2,5,6]
	test_val = False
	next_iter_flag = False
	counter = 0 
	while curretn_tweet[2][:-3] < '1483228860':
		curretn_tweet = next(iter_tweets)
	for row in list(csv_file)[1:]:
		#print(row[0], curretn_tweet[2][:-3])
		if row[0] < str(curretn_tweet[2])[:-2]:
			tweets_norm.append(['None','0',row[0]])
		else:
			tweets_norm.append([curretn_tweet[0],curretn_tweet[1],row[0]])
			try:
				curretn_tweet = next(iter_tweets)
				counter += 1
			except StopIteration:
				curretn_tweet = ['None',0,'9999999999']
		print(counter)




with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/twitter_norm_2017_2020.csv', 'w', newline='') as obj:
	fill_file = writer(obj)
	fill_file.writerows(tweets_norm)

