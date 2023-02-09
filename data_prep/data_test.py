from csv import reader
from csv import writer
import matplotlib.pyplot as plt
import numpy as np


returns = []
step = 0
total_list = []
total = 0
with open('/Users/danilkutny/Desktop/crypto_trading/rl_agnt_000001/r_histor_7300.csv', 'r', newline='') as obj:
	fill_file = reader(obj)
	for row in list(fill_file):
		returns.append(float(row[0]))
		total+=float(row[0])
		total_list.append(total)
total_bh_list = []
total_bh = 0
with open('/Users/danilkutny/Desktop/crypto_trading/rl_agnt_000001/bh_history_7300.csv', 'r', newline='') as obj:
	fill_file = reader(obj)
	for row in list(fill_file):
		total_bh+=float(row[0])
		total_bh_list.append(total_bh)


plot1 = plt.figure(1)
plt.plot(total_list)
plot1 = plt.figure(2)
plt.plot(total_bh_list)
plt.show()