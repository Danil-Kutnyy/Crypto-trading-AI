from csv import reader
from csv import writer
full_list = [['Unix Timestamp', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
'''
with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/gemini_min_2016_2020.csv', 'r',newline='') as obj:
	csv_file = reader(obj)
	for row in list(csv_file)[1:]:
		print(row)
'''
with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/bitstamp_min_2016_2020.csv', 'r', newline='') as obj:
	csv_file = reader(obj)
	counter_10 = 0
	start = 966.34
	end = 0
	high = 0
	low = 999999999999999
	vol = 0
	vol_2 = 0
	cheker = False
	for row in list(csv_file)[1:]:
		'''
		if row[0] == '1542240000000':
			cheker = True
		if cheker == True:	
		'''
		end = float(row[5])
		vol += float(row[6])
		vol_2 += float(row[7])
		if float(row[3]) > float(high):
			high = float(row[3])
		if float(row[4]) < float(low):
			low = float(row[4])
		if counter_10 % 1440 == 0 and counter_10 != 0: 
			full_list.append([row[0],row[1],start,high,low,end,vol,vol_2])
			start = float(row[5])
			high = 0
			low = 999999999999999
			vol = 0
			vol_2 = 0
		counter_10 += 1

for i in full_list:
	print(i)


with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/bitstamp_days_2016_2020.csv', 'w', newline='') as obj:
	fill_file = writer(obj)
	fill_file.writerows(full_list)

