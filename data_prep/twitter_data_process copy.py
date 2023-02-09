from csv import reader
from csv import writer
new_data = []
with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/twitter_17_20_2.csv', 'r', newline='') as obj:
	fill_file = reader(obj)
	for row in list(fill_file):
		print(row)
		#new_data.append(row)
'''
with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/twitter_17_20_2.csv', 'w', newline='') as obj:
	fill_file = writer(obj, delimiter=';')
	fill_file.writerows(new_data)
'''