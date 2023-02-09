#general
import os
import numpy as np
from crypto_trade_env import btstamp_exhange
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import csv

#bert sepcific
import os
import shutil
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

eps = np.finfo(np.float32).eps.item()
backday_window = 180
hours_window = 168
minutes_10_window = 144
minutes_window = 120
time_fetures = 10
learning_rate = 0.000001
lr_str = '000001'
gamma = 1.0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#bert settings
bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8' 

map_name_to_handle = {
	'bert_en_uncased_L-12_H-768_A-12':
		'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
	'bert_en_cased_L-12_H-768_A-12':
		'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
	'bert_multi_cased_L-12_H-768_A-12':
		'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
	'small_bert/bert_en_uncased_L-2_H-128_A-2':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
	'small_bert/bert_en_uncased_L-2_H-256_A-4':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
	'small_bert/bert_en_uncased_L-2_H-512_A-8':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
	'small_bert/bert_en_uncased_L-2_H-768_A-12':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
	'small_bert/bert_en_uncased_L-4_H-128_A-2':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
	'small_bert/bert_en_uncased_L-4_H-256_A-4':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
	'small_bert/bert_en_uncased_L-4_H-512_A-8':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
	'small_bert/bert_en_uncased_L-4_H-768_A-12':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
	'small_bert/bert_en_uncased_L-6_H-128_A-2':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
	'small_bert/bert_en_uncased_L-6_H-256_A-4':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
	'small_bert/bert_en_uncased_L-6_H-512_A-8':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
	'small_bert/bert_en_uncased_L-6_H-768_A-12':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
	'small_bert/bert_en_uncased_L-8_H-128_A-2':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
	'small_bert/bert_en_uncased_L-8_H-256_A-4':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
	'small_bert/bert_en_uncased_L-8_H-512_A-8':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
	'small_bert/bert_en_uncased_L-8_H-768_A-12':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
	'small_bert/bert_en_uncased_L-10_H-128_A-2':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
	'small_bert/bert_en_uncased_L-10_H-256_A-4':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
	'small_bert/bert_en_uncased_L-10_H-512_A-8':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
	'small_bert/bert_en_uncased_L-10_H-768_A-12':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
	'small_bert/bert_en_uncased_L-12_H-128_A-2':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
	'small_bert/bert_en_uncased_L-12_H-256_A-4':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
	'small_bert/bert_en_uncased_L-12_H-512_A-8':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
	'small_bert/bert_en_uncased_L-12_H-768_A-12':
		'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
	'albert_en_base':
		'https://tfhub.dev/tensorflow/albert_en_base/2',
	'electra_small':
		'https://tfhub.dev/google/electra_small/2',
	'electra_base':
		'https://tfhub.dev/google/electra_base/2',
	'experts_pubmed':
		'https://tfhub.dev/google/experts/bert/pubmed/2',
	'experts_wiki_books':
		'https://tfhub.dev/google/experts/bert/wiki_books/2',
	'talking-heads_base':
		'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}

map_model_to_preprocess = {
	'bert_en_uncased_L-12_H-768_A-12':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'bert_en_cased_L-12_H-768_A-12':
		'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
	'small_bert/bert_en_uncased_L-2_H-128_A-2':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-2_H-256_A-4':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-2_H-512_A-8':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-2_H-768_A-12':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-4_H-128_A-2':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-4_H-256_A-4':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-4_H-512_A-8':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-4_H-768_A-12':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-6_H-128_A-2':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-6_H-256_A-4':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-6_H-512_A-8':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-6_H-768_A-12':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-8_H-128_A-2':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-8_H-256_A-4':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-8_H-512_A-8':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-8_H-768_A-12':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-10_H-128_A-2':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-10_H-256_A-4':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-10_H-512_A-8':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-10_H-768_A-12':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-12_H-128_A-2':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-12_H-256_A-4':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-12_H-512_A-8':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'small_bert/bert_en_uncased_L-12_H-768_A-12':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'bert_multi_cased_L-12_H-768_A-12':
		'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
	'albert_en_base':
		'https://tfhub.dev/tensorflow/albert_en_preprocess/3',
	'electra_small':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'electra_base':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'experts_pubmed':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'experts_wiki_books':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
	'talking-heads_base':
		'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)


'''
technical_fetures_extactor = tf.keras.models.load_model('/Users/danilkutny/Desktop/crypto_trading/raw_dataset/lr_00035/model_3915')
#Crypto model starts here---------------------------------------------------------------------------------------------------------------

#inputs
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
#technical_fetures_extactor.input

#bert model
encoder_inputs = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')(text_input)
bert_layer = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')(encoder_inputs)
bert_outputs = bert_layer['pooled_output']

#tecnical features
technical_fetures_layer = technical_fetures_extactor.layers[-3].output

all_features = tf.keras.layers.Concatenate()([technical_fetures_layer,bert_outputs])#output = (1034)

pre_lstmn_dense = layers.Dense(1024, activation='relu')(all_features)

lstm_preprocess = layers.Reshape(target_shape=(1, 1024))(pre_lstmn_dense)
lstm_layer = layers.LSTM(1024, stateful=True)(lstm_preprocess)
lstm_preprocess_2 = layers.Reshape(target_shape=(1, 1024))(lstm_layer)
lstm_2_layer = layers.LSTM(512, stateful=True)(lstm_preprocess_2)

fully_conected = layers.Dense(256, activation='relu')(lstm_2_layer)

#prediction
acion = layers.Dense(2,activation='softmax')(fully_conected)
critic = layers.Dense(1)(fully_conected)

#Crypto model ends here---------------------------------------------------------------------------------------------------------------

#technical_fetures_layer.trainable = False


model = keras.Model(inputs=[ 
	technical_fetures_extactor.input,
	text_input],
	outputs=[acion, critic])
'''

model = tf.keras.models.load_model('/Users/danilkutny/Desktop/crypto_trading/rl_agnt_full_000001/model_700')


model.layers[-10].trainable = True
model.layers[-14].trainable = True

model.summary()

print('learning_rate:',learning_rate)

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
huber_loss = keras.losses.Huber()
'''
epochs = 1
steps_per_epoch = np.array(256)
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)
init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
										  num_train_steps=num_train_steps,
										  num_warmup_steps=num_warmup_steps,
										  optimizer_type='adamw')
'''
bh_env = btstamp_exhange()
dummy, dummy, dummy, dummy, dummy, dummy = bh_env.start()


mse_loss = tf.losses.MeanSquaredError()

env = btstamp_exhange()
r_history = []
b_h_history = []
runing_reward_rl = 0.0
runing_reward_bh = 0.0

min_dat, min10_dat, hrs_dat, dys_dat, twit_dat, time_dat = env.start()
for train_i in range(15500):
	if train_i<701:
		for step_i in range(120):
			min_dat, min10_dat, hrs_dat, dys_dat, twit_dat, time_dat, reward = env.step(0)
			dummy, dummy, dummy, dummy, dummy, dummy, bh_reward = bh_env.step(0)
		if train_i % 100==0:
			print('Looping! #',train_i)
	else:
		if train_i == 701:
			print('finaly training!')
		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(model.trainable_variables)
			action_probs_history = []
			critic_value_history = []
			rewards_history = []

			episode_res = 0
			episode_bh_res = 0
			for step_i in range(120):

				#min_dat, min10_dat, hrs_dat, dys_dat, twit_dat, time_dat, y_true = env.step(2)

				min_dat = tf.convert_to_tensor(min_dat)
				min_dat = tf.expand_dims(min_dat, 0)

				min10_dat = tf.convert_to_tensor(min10_dat)
				min10_dat = tf.expand_dims(min10_dat, 0)

				hrs_dat = tf.convert_to_tensor(hrs_dat)
				hrs_dat = tf.expand_dims(hrs_dat, 0)

				dys_dat = tf.convert_to_tensor(dys_dat)
				dys_dat = tf.expand_dims(dys_dat, 0)

				time_dat = tf.convert_to_tensor(time_dat)
				time_dat = tf.expand_dims(time_dat, 0)

				twit_dat = tf.convert_to_tensor(twit_dat)
				twit_dat = tf.expand_dims(twit_dat, 0)

				action_probs, critic_value = model([min_dat, min10_dat, hrs_dat, dys_dat, time_dat, twit_dat])

				critic_value_history.append(critic_value[0, 0])
				action = np.random.choice(2, p=np.squeeze(action_probs))
				action_probs_history.append(tf.math.log(action_probs[0, action]))

				min_dat, min10_dat, hrs_dat, dys_dat, twit_dat, time_dat, reward = env.step(action)
				dummy, dummy, dummy, dummy, dummy, dummy, bh_reward = bh_env.step(1)

				episode_res+=reward
				episode_bh_res+=bh_reward
				
				rewards_history.append(reward)


			returns = []
			discounted_sum = 0
			for r in rewards_history[::-1]:
				discounted_sum = r + gamma * discounted_sum
				returns.insert(0, discounted_sum)

			returns = np.array(returns)
			returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
			returns = returns.tolist()

			history = zip(action_probs_history, critic_value_history, returns)
			actor_losses = []
			critic_losses = []
			for log_prob, value, ret in history:
				# At this point in history, the critic estimated that we would get a
				# total reward = `value` in the future. We took an action with log probability
				# of `log_prob` and ended up recieving a total reward = `ret`.
				# The actor must be updated so that it predicts an action that leads to
				# high rewards (compared to critic's estimate) with high probability.
				diff = ret - value
				actor_losses.append(-log_prob * diff)  # actor loss

				# The critic must be updated so that it predicts a better estimate of
				# the future rewards.
				critic_losses.append(
					huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
				)

			loss_value = sum(actor_losses) + sum(critic_losses)
			grads = tape.gradient(loss_value, model.trainable_variables)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
		
		
		#print('episode_res:',episode_res)
		#print('episode_res:',episode_res)
		b_h_history.append(episode_bh_res)
		r_history.append(episode_res)
		#print('episode_bh_res:',episode_bh_res)
		episode_res_str = "{:.2f}".format(episode_res)
		episode_bh_res_str = "{:.2f}".format(episode_bh_res)

		runing_reward_rl = 0.05 * episode_res + (1 - 0.05) * runing_reward_rl
		#print('runing_reward_rl:',runing_reward_rl)
		runing_reward_bh = 0.05 * episode_bh_res + (1 - 0.05) * runing_reward_bh
		
		if train_i % 100==0:
			model.save('/Users/danilkutny/Desktop/crypto_trading/rl_agnt_full_{}/model_{}'.format(lr_str,train_i))
			print('#{}'.format(train_i),'rl and bh current:',episode_res_str,episode_bh_res_str,'| rl and bh running',runing_reward_rl, runing_reward_bh, '| %:', episode_res/(episode_bh_res+eps))
			#print('#{}'.format(train_i))
			#print('rl and bh current:', episode_res_str, episode_bh_res)
			#print('| rl and bh running',episode_bh_res_str, runing_reward_rl)
			#print('| %:', episode_res/(episode_bh_res+eps))

			if train_i != 0:
				np.savetxt('/Users/danilkutny/Desktop/crypto_trading/rl_agnt_full_{}/bh_history_{}.csv'.format(lr_str,train_i), np.squeeze(np.array(b_h_history)), delimiter=",")
				np.savetxt('/Users/danilkutny/Desktop/crypto_trading/rl_agnt_full_{}/r_histor_{}.csv'.format(lr_str,train_i), np.squeeze(np.array(r_history)), delimiter=",")

	























