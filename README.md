# Crypto-trading-AI
AI trained to trade cryptocurrency

This is my project to test, how modern neural network can trade crytocurrency.



Data for mode training can be ascces with google drive:
If consist of market data from bitstamp crypto-exchange, from 2017 up to 2020.
  Bitstamp market data form:
  Unix Timestamp | Open	| High | Low | Close | Volume
Lowest resolution of trnsation is minutes.

Also, it contains twitter data, consisted of famous crypto-accounts and all their tweets for correspond date, adn also how mane likes thistweet collects.
Form:
  Unix Timestamp | tweet content	| # of likes



This data was preporated with custom python code, contained in data_prep folder



Then, this data is used to simulate market environment. As my Ai bot trained to trade, rather then predict, i had to simualte a market
for it to play and learn. Basics, of how this market evonoment works is described in my thesis. 
This theisis is written for my another project - stock trading bot, but it has many simularities, like environemtn and neural network strucutre and
learning.


Market enviroonmet represnts an object - market. It has only one method - step, which accept 0 or 1 as a parameter.
To play on the market - you have to choose, buy or sell cryptoasset. (1 or 0).
After desision is made, step method return result for the next minute of market data and calulate your reward.
In this way, Neural network can make action (buy or sell) and immediately get response (reward), of how well it performed, 
and also gets new market data(Open	| High | Low | Close | Volume & twiite_content | likes) to make next action.
AI agent also get information about its account. 
It starts with 1 dollar in tis account and continues to play for 4 years in a row.


Trianing of the neural network is happening in the RL_train.py. Policy gradients is used as a reinforment leaning policy




Model_7400 files contains tensorflow trained model, which was trained for 4 years of per-minute data



Trading bot results contains 3 model result with different parameter: -30, 30 adn 50.
acc_histor_7300.csv - contains information of accout balance during lasst training session(one trainign session = 120 minutes)
bh_history_7300.csv - contains information of accout balance during lasst training session, if the agent using buy&hold strategy
r_histor_7300.csv - contains information of model reuturn(decrese or increse in total capital) during last traning session
