from csv import reader
from csv import writer
from datetime import datetime
y2016 = None
y2017 = None
y2018 = None
y2019 = None
y2020 = None
full_list = []

with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/raw_bitstamp/Bitstamp_BTCUSD_2017_minute.csv', 'r',newline='') as obj:
	csv_file = reader(obj)
	for row in list(csv_file)[1:]:
		full_list.append([item for item in row if row.index(item)!= 2])
		break
with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/raw_bitstamp/Bitstamp_BTCUSD_2017_minute.csv', 'r',newline='') as obj:
	csv_file = reader(obj)
	for row2 in reversed(list(csv_file)[2:]):
		full_list.append([int(row2[0]),row2[1],float(row2[3]),float(row2[4]),float(row2[5]),float(row2[6]),float(row2[7]),float(row2[8])])
		#full_list.append([item for item in row2 if row2.index(item)!=2])

with open('/Users/danilkutny/Desktop/Bitstamp_BTCUSD_2018_minute.csv', 'r',newline='') as obj:
	csv_file = reader(obj,delimiter=';')
	for row2 in reversed(list(csv_file)[2:]):
		full_list.append([int(row2[0]),row2[1],float(row2[3]),float(row2[4]),float(row2[5]),float(row2[6]),float(row2[7]),float(row2[8])])

with open('/Users/danilkutny/Desktop/Bitstamp_BTCUSD_2019_minute.csv', 'r',newline='') as obj:
	csv_file = reader(obj,delimiter=';')
	for row2 in reversed(list(csv_file)[2:]):
		full_list.append([int(row2[0]),row2[1],float(row2[3]),float(row2[4]),float(row2[5]),float(row2[6]),float(row2[7]),float(row2[8])])

with open('/Users/danilkutny/Desktop/Bitstamp_BTCUSD_2020_minute.csv', 'r',newline='') as obj:
	csv_file = reader(obj,delimiter=';')
	for row2 in reversed(list(csv_file)[2:]):
		full_list.append([int(row2[0]),row2[1],float(row2[3]),float(row2[4]),float(row2[5]),float(row2[6]),float(row2[7]),float(row2[8])])
		if row2[0] == '1606089540':
			break

with open('/Users/danilkutny/Downloads/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv', 'r',newline='') as obj:
	csv_file = reader(obj)
	chek_2 = False
	for row2 in list(csv_file)[2:]:
		#print(row2)
		if row2[0] == '1606089600':
			chek_2 = True
		if chek_2 == True:
			full_list.append([int(row2[0]),str(datetime.utcfromtimestamp(int(row2[0])).strftime('%Y-%m-%d %H:%M:%S')),float(row2[1]),float(row2[2]),float(row2[3]),float(row2[4]),float(row2[5]),float(row2[6])])
		if row2[0] == '1608508740':
			break

with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/raw_bitstamp/Bitstamp_BTCUSD_2020_minute.csv', 'r',newline='') as obj:
	csv_file = reader(obj)
	chek = False
	for row2 in reversed(list(csv_file)[2:]):
		if row2[0] == '1608508800':
			chek = True
		if chek == True:
			full_list.append([int(row2[0]),row2[1],float(row2[3]),float(row2[4]),float(row2[5]),float(row2[6]),float(row2[7]),float(row2[8])])

'''
with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/Bitstamp_BTCUSD_2018_minute.csv', 'r',newline='') as obj:
	csv_file = reader(obj)
	for row2 in reversed(list(csv_file)[2:]):
		full_list.append([item for item in row2 if row2.index(item)!= 2])

with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/gemini_BTCUSD_2018_1min.csv', 'r',newline='') as obj:
	csv_file = reader(obj)
	for row2 in reversed(list(csv_file)[2:]):
		full_list.append([item for item in row2 if row2.index(item)!= 2])

with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/gemini_BTCUSD_2019_1min.csv', 'r',newline='') as obj:
	csv_file = reader(obj)
	for row2 in reversed(list(csv_file)[2:]):
		full_list.append([item for item in row2 if row2.index(item)!= 2])

with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/gemini_BTCUSD_2020_1min.csv', 'r',newline='') as obj:
	csv_file = reader(obj)
	for row2 in reversed(list(csv_file)[2:]):
		full_list.append([item for item in row2 if row2.index(item)!= 2])
'''

for i in full_list:
	print(i)


with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/bitstamp_min_2016_2020.csv', 'w', newline='') as obj:
	fill_file = writer(obj)
	fill_file.writerows(full_list)

