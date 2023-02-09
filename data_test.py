from csv import reader
from csv import writer
import matplotlib.pyplot as plt
import numpy as np


backday_window = 180
hours_window = 168
minutes_10_window = 144
minutes_window = 120
'''
start_minutess = (backday_window * 24 * 60) - minutes_window + 1
minnutes_data = np.genfromtxt('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/btmp_17_20_m.csv', delimiter=',')[start_minutess:]
ietrable = iter(minnutes_data)
counter = 0 
for i in range(14500):
	for b in range(120):
		a = next(ietrable)
		counter +=1
		if counter > 900000:
			print(a[0])
'''



returns = []
step = 0
total_list = []
total = 0
with open('/Users/danilkutny/Desktop/crypto_trading/test_max_30/r_histor_7300.csv', 'r', newline='') as obj:
	fill_file = reader(obj)
	for row in list(fill_file):
		total+=float(row[0])
		total_list.append(total)
total_bh_list = []
total_bh = 0
with open('/Users/danilkutny/Desktop/crypto_trading/test_max_30/bh_history_7300.csv', 'r', newline='') as obj:
	fill_file = reader(obj)
	for row in list(fill_file):
		total_bh+=float(row[0])
		total_bh_list.append(total_bh)
total_list_3 = []
total_3 = 0

with open('/Users/danilkutny/Desktop/crypto_trading/test_max_30/acc_histor_7300.csv', 'r', newline='') as obj:
	fill_file = reader(obj)
	for row in list(fill_file):
		total_3+=float(row[0])
		total_list_3.append(float(row[0]))



plot1 = plt.figure(1)
plt.plot(total_list)
plot1 = plt.figure(2)
plt.plot(total_bh_list)
plot1 = plt.figure(3)
plt.plot(total_list_3)
plt.show()

'''
length = [i for i in range(1,len(total_bh_list)+1)]
plot1 = plt.figure(1)
ax1 = plot1.add_axes([0,0,1,1])
ax1.bar(length, total_list)
plot2 = plt.figure(2)
ax2 = plot2.add_axes([0,0,1,1])
ax2.bar(length, total_bh_list)
plt.show()
'''