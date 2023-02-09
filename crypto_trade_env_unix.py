import csv
from datetime import datetime
import numpy as np
from numpy import genfromtxt
import random
import math
import csv

backday_window = 180
hours_window = 168
minutes_10_window = 144
minutes_window = 120
def sigmoid_500(x):
  return 1 / (1 + math.exp(-x*700))

'''
minnutes_data = genfromtxt('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/bitstamp_min_2016_2020.csv', delimiter=',')
minnutes_10_data = genfromtxt('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/bitstamp_10min_2016_2020.csv', delimiter=',')
hours_data = genfromtxt('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/bitstamp_hour_2016_2020.csv', delimiter=',')
days_data = genfromtxt('/Users/danilkutny/Desktop/crypto_trading/bitstamp_days_2016_2020.csv', delimiter=',')
twitter_data = genfromtxt('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/twitter_norm_2017_2020.csv', delimiter=',')
'''
start_minutess = (backday_window * 24 * 60) - minutes_window + 1
start_10_minutes = (backday_window * 24 * 6) - minutes_10_window 
start_1hours = (backday_window * 24) - hours_window 
start_days = backday_window

days_data = genfromtxt('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/btmp_17_20_d.csv', delimiter=',')
minnutes_data = genfromtxt('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/btmp_17_20_m.csv', delimiter=',')[start_minutess:] #add ,2:7 inside index to remoove date and unix
minnutes_10_data = genfromtxt('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/btmp_17_20_10m.csv', delimiter=',')[start_10_minutes:]
hours_data = genfromtxt('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/btmp_17_20_h.csv', delimiter=',')[start_1hours:]
time_data = genfromtxt('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/time_encoded_16_20.csv', delimiter=',')[start_minutess+minutes_window-1:]
#twitter_data = genfromtxt('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/twitter_17_20_2.csv', delimiter=';')[start_minutess+minutes_window:]

with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/twitter_17_20.csv', 'r', newline='') as obj:
	fill_file = csv.reader(obj)
	twitter_data = list(fill_file)[start_minutess+minutes_window-1:]

minutes_vis_data = minnutes_data[:minutes_window]
minutes_10_vis_data = minnutes_10_data[:minutes_10_window]
hours_vis_data = hours_data[:hours_window]
days_vis_data = days_data[:backday_window]

twitter_iter = iter(twitter_data)
twitter_vis_data = next(twitter_iter)
time_iter = iter(time_data)
time_vis_data = next(time_iter)
'''
print(minutes_vis_data[-1][0])
print(minutes_10_vis_data[-1][0])
print(hours_vis_data[-1][0])
print(days_vis_data[-1][0])
print(twitter_vis_data)
print(time_vis_data)
'''



minutes_data = minnutes_data[minutes_window:]
minutes_10_data = minnutes_10_data[minutes_10_window:]
hours_data = hours_data[hours_window:]
days_data = days_data[backday_window:]

'''
print(minutes_data[0,0])
print(minutes_10_data[0,0])
print(hours_data[0,0])
print(days_data[0,0])

'''
min_iter = iter(minutes_data)
min_10_iter = iter(minnutes_10_data)
hours_iter = iter(hours_data)
days_iter = iter(days_data)
days_iter = iter(days_data)
class btstamp_exhange():

	def __init__(self):
		self.current_price = minutes_vis_data[-1][5]
		self.minutes_vis_data= minutes_vis_data
		self.minutes_10_vis_data = minutes_10_vis_data
		self.hours_vis_data = hours_vis_data
		self.days_vis_data = days_vis_data
		self.twitter_vis_data = twitter_vis_data
		self.time_vis_data = time_vis_data
		self.total_steps = 1
		self.action = 0

	def start(self):
		return self.minutes_vis_data, self.minutes_10_vis_data, self.hours_vis_data, self.days_vis_data, self.twitter_vis_data, self.time_vis_data

	def step(self, action):
		new_values = next(min_iter)
		if action == 2:
			self.reward = sigmoid_500((new_values[5] - self.current_price)/new_values[5])
		if self.action == 1:
			self.reward = (new_values[5] - self.current_price)
		if action == 0:
			self.reward = 0
		self.action = action
		self.current_price = new_values[5]
		
		self.minutes_vis_data = np.vstack((self.minutes_vis_data[1:], new_values))
		self.twitter_vis_data = next(twitter_iter)
		self.time_vis_data = next(time_iter)
		
		if self.total_steps % 10 == 0:
			self.minutes_10_vis_data = np.vstack( (self.minutes_10_vis_data[1:], next(min_10_iter)) )
		if self.total_steps % 60  == 0:
			self.hours_vis_data = np.vstack( (self.hours_vis_data[1:], next(hours_iter)) )
		if self.total_steps % 1440 == 0:
			#print('_______________NEW DAY______________')
			self.days_vis_data = np.vstack( (self.days_vis_data[1:], next(days_iter)) )
		self.total_steps += 1
		return self.minutes_vis_data, self.minutes_10_vis_data, self.hours_vis_data, self.days_vis_data, self.twitter_vis_data, self.time_vis_data, self.reward


env = btstamp_exhange()
min_wind, min_10_wind, hour_wind, days_wind, tweet, time = env.start()


for i in range(7400):
	for i in range(120):
		min_wind, min_10_wind, hour_wind, days_wind, tweet, time, y_pred = env.step(2)
		#print(str(min_wind[-1][0])[:-2])

		print(datetime.utcfromtimestamp(int(str(min_wind[-1][0])[:-2])).strftime('%Y-%m-%d %H:%M:%S'))





