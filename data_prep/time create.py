import csv
import random
import os
from datetime import datetime
from datetime import timedelta
from datetime import date
import math
import numpy as np
import calendar
import sys
np.set_printoptions(threshold=sys.maxsize)
start = datetime.strptime('2017-01-01', '%Y-%m-%d')
ts = 1493173440
with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/bitstamp_min_2016_2020.csv', 'r', newline='') as obj:
	csv_file = csv.reader(obj)
	all_data = list(csv_file)[1:]


def encoded_date(date_datetime, unix):
	date_datetime = datetime.strptime(date_datetime, '%Y-%m-%d %H:%M:%S')
	h_1 = (np.sin(int(str(date_datetime)[14:16]) * 2 * np.pi / 59)+1)/2 
	h_2 = (np.cos(int(str(date_datetime)[14:16]) * 2 * np.pi / 59)+1)/2 
	
	d_1 = (np.sin(int(str(date_datetime)[11:13]) * 2 * np.pi / 23)+1)/2 
	d_2 = (np.cos(int(str(date_datetime)[11:13]) * 2 * np.pi / 23)+1)/2 
	
	w_1 = (np.sin(date_datetime.weekday() * 2 * np.pi / 7)+1)/2 
	w_2 = (np.cos(date_datetime.weekday() * 2 * np.pi / 7)+1)/2  
	
	m_1 = (np.sin(date_datetime.day * 2 * np.pi / calendar.monthrange(date_datetime.year, date_datetime.month)[1])+1)/2 
	m_2 = (np.cos(date_datetime.day * 2 * np.pi / calendar.monthrange(date_datetime.year, date_datetime.month)[1])+1)/2 
	
	y_1 = (np.sin(date_datetime.timetuple().tm_yday * 2 * np.pi / 366)+1)/2
	y_2 = (np.cos(date_datetime.timetuple().tm_yday * 2 * np.pi / 366)+1)/2
	return [unix, y_1, y_2, m_1, m_2, w_1, w_2, d_1, d_2, h_1, h_2]

def encoded_date_2(date_datetime, unix):
	date_datetime = datetime.strptime(date_datetime, '%d.%m.%Y %H:%M:%S')
	h_1 = (np.sin(int(str(date_datetime)[14:16]) * 2 * np.pi / 59)+1)/2 
	h_2 = (np.cos(int(str(date_datetime)[14:16]) * 2 * np.pi / 59)+1)/2 
	
	d_1 = (np.sin(int(str(date_datetime)[11:13]) * 2 * np.pi / 23)+1)/2 
	d_2 = (np.cos(int(str(date_datetime)[11:13]) * 2 * np.pi / 23)+1)/2 
	
	w_1 = (np.sin(date_datetime.weekday() * 2 * np.pi / 7)+1)/2 
	w_2 = (np.cos(date_datetime.weekday() * 2 * np.pi / 7)+1)/2  
	
	m_1 = (np.sin(date_datetime.day * 2 * np.pi / calendar.monthrange(date_datetime.year, date_datetime.month)[1])+1)/2 
	m_2 = (np.cos(date_datetime.day * 2 * np.pi / calendar.monthrange(date_datetime.year, date_datetime.month)[1])+1)/2 
	
	y_1 = (np.sin(date_datetime.timetuple().tm_yday * 2 * np.pi / 366)+1)/2
	y_2 = (np.cos(date_datetime.timetuple().tm_yday * 2 * np.pi / 366)+1)/2
	return [unix, y_1, y_2, m_1, m_2, w_1, w_2, d_1, d_2, h_1, h_2]

def encoded_date_2(date_datetime, unix):
	date_datetime = datetime.strptime(date_datetime, '%d.%m.%Y %H:%M:%S')
	h_1 = (np.sin(int(str(date_datetime)[14:16]) * 2 * np.pi / 59)+1)/2 
	h_2 = (np.cos(int(str(date_datetime)[14:16]) * 2 * np.pi / 59)+1)/2 
	
	d_1 = (np.sin(int(str(date_datetime)[11:13]) * 2 * np.pi / 23)+1)/2 
	d_2 = (np.cos(int(str(date_datetime)[11:13]) * 2 * np.pi / 23)+1)/2 
	
	w_1 = (np.sin(date_datetime.weekday() * 2 * np.pi / 7)+1)/2 
	w_2 = (np.cos(date_datetime.weekday() * 2 * np.pi / 7)+1)/2  
	
	m_1 = (np.sin(date_datetime.day * 2 * np.pi / calendar.monthrange(date_datetime.year, date_datetime.month)[1])+1)/2 
	m_2 = (np.cos(date_datetime.day * 2 * np.pi / calendar.monthrange(date_datetime.year, date_datetime.month)[1])+1)/2 
	
	y_1 = (np.sin(date_datetime.timetuple().tm_yday * 2 * np.pi / 366)+1)/2
	y_2 = (np.cos(date_datetime.timetuple().tm_yday * 2 * np.pi / 366)+1)/2
	return [unix, y_1, y_2, m_1, m_2, w_1, w_2, d_1, d_2, h_1, h_2]

date_csv = []
cheker = False
for dat in all_data:
	#print(dat)
	if dat[1] == '01.01.2018 00:00:00':
		date_csv.append(encoded_date_2(dat[1], dat[0]))
	#if cheker == True:
	#	date_csv.append(encoded_date_2(dat[1]))
	else:
		date_csv.append(encoded_date(dat[1], dat[0]))
'''
for a, b in zip(date_csv, all_data):
	print(b[1],a)
'''

with open('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/time_encoded_2017_2020.csv', 'w', newline='') as obj:
	fill_file = csv.writer(obj)
	fill_file.writerows(date_csv)

	