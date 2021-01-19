#!/usr/bin/env python3


from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 

from sklearn import metrics
import time
from pprint import pprint

# univariate lstm example
from numpy import array
from keras.models import Sequential, Model
from keras.layers import LSTM, ConvLSTM2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
# from keras.layers import 

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
from keras.models import load_model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def plot_time_series(time, values, label):
  plt.figure(figsize=(10,6))
  plt.plot(time, values)
  plt.xlabel("Time", fontsize=20)
  plt.ylabel("Value", fontsize=20)
  plt.title(label, fontsize=20)
  plt.grid(True)

def convex_curve(len_, a=1, b=0, t=0):
  x = np.arange(len_, dtype=np.float128)
  y = np.zeros(len_, dtype=np.float128)
  y = np.where(x>0, -a*np.log(x) + b, b)
  return y

def concave_curve(len_, a=1, b=0, t=0):
  x = np.arange(len_, dtype=np.float128)
  y = np.zeros(len_, dtype=np.float128)
  y = np.where(x>0, a*np.log(x) + b, b)
  return y

def big_event(len_, subs_=5):
  time = np.arange(len_/subs_)
  values = np.where(time < 10, time**3, (time-9)**2)
  # Repeat the pattern 5 times
  seasonal = []
  for i in range(subs_):
    for j in range(len_//subs_):
      seasonal.append(values[j])# Plot

  noise = np.random.randn(len_)*100
  seasonal += noise
  time_seasonal = np.arange(len_)

  seasonal_upward = seasonal + np.arange(len_)*10

  time_seasonal = np.arange(len_)
  big_event = np.zeros(len_)
  big_event[-len_//subs_:] = np.arange(len_//subs_)*-len_//subs_
  non_stationary = seasonal_upward + big_event
  time_seasonal = np.arange(len_)
  return time_seasonal, non_stationary
# len_ = 250
# subs_=4
# time_seasonal, non_stationary = big_event(len_, 5)
# plot_time_series(time_seasonal, non_stationary, label="Non-stationary Time Series")

def sine_wave(len_, a, b):
  l = np.linspace(0, 1, len_)
  y = a*np.sin(2 * np.pi * b * l)
  return y

def trend_up(len_, a, b):
  x = np.arange(len_)
  y = np.where(x>=0, a * x + b, a * x + b)
  return y

def trend_down(len_, a, b):
  x = np.arange(len_)
  y = np.where(x>=0, -a * x + b, -a * x + b)
  return y

def constant(len_, a):
  x = np.arange(len_)
  y = np.where(x>=0, a, a)
  return y

def cn(n):
  c = y*np.exp(-1j*2*n*np.pi*time/period)
  return c.sum()/c.size

def f(x, Nh):
  f = np.array([2*cn(i)*np.exp(1j*2*i*np.pi*x/period) for i in range(1,Nh+1)])
  return f.sum()

def complex_test():
  a = 1
  b = 3
  len_ = 100
  # plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

  con1 = sine_wave(len_+50, a+20, 2)
  con2 = convex_curve(len_+50, 2, 1)
  con3 = trend_up(len_+100, 0.2, 2)
  con4 = trend_down(len_+50, 0.1, 8)
  con5 = concave_curve(len_, 10, 5)
  con6 = convex_curve(len_+50, 1, 2)
  # con7 = convex_curve(len_, 2, 3)
  # con8 = sine_wave(len_, a+2, 2)
  # con9 = sine_wave(len_, a+87, 4)
  # con10 = sine_wave(len_, a+12, 8)



  complex_ = np.concatenate([con1, con2, con3, con4, con5, con6]) #, con7, con8, con9, con10])
  return complex_

def complex_sel_test():
  a = 1
  b = 3
  len_ = 100
  # plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

  # con1 = sine_wave(len_+50, a+10, 5)
  con2 = convex_curve(len_+150, 2, 10)
  # con3 = trend_up(len_+100, 0.2, 3)
  con4 = trend_down(len_+150, 0.2, 8)
  con5 = concave_curve(len_+150, 8, 6)
  con6 = concave_curve(len_+10, 10, 3)
  # con6 = convex_curve(len_+50, 2, 3)
  # con7 = convex_curve(len_, 2, 3)
  # con8 = sine_wave(len_, a+2, 2)
  # con9 = sine_wave(len_, a+87, 4)
  # con10 = sine_wave(len_, a+12, 8)



  complex_test = np.concatenate([con2, con4, con5, con6]) #, con7, con8, con9, con10])
  return complex_test


def complex_sel():
  a = 1
  b = 3
  len_ = 100
  # plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

  # con1 = sine_wave(len_+50, a+10, 5)
  con2 = convex_curve(len_+150, 2, 10)
  # con3 = trend_up(len_+100, 0.2, 3)
  con4 = trend_down(len_+150, 0.2, 8)
  con5 = concave_curve(len_+150, 8, 6)
  # con6 = convex_curve(len_+50, 2, 3)
  # con7 = convex_curve(len_, 2, 3)
  # con8 = sine_wave(len_, a+2, 2)
  # con9 = sine_wave(len_, a+87, 4)
  # con10 = sine_wave(len_, a+12, 8)



  complex_t = np.concatenate([con2, con4, con5]) #, con7, con8, con9, con10])
  return complex_t

def complex_():
  a = 1
  b = 3
  len_ = 100
  # plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

  con1 = sine_wave(len_, a+10, 5)
  con2 = convex_curve(len_, 2, 10)
  con3 = trend_up(len_, 0.2, 3)
  con4 = trend_down(len_, 0.2, 8)
  con5 = concave_curve(len_, 8, 6)
  con6 = convex_curve(len_, 2, 3)
  con7 = convex_curve(len_, 2, 3)
  con8 = sine_wave(len_, a+2, 2)
  con9 = sine_wave(len_, a+87, 4)
  con10 = sine_wave(len_, a+12, 8)



  complex_t = np.concatenate([con1, con2, con3, con4, con5, con6, con7, con8, con9, con10])
  return complex_t

def sin():
  a = 1
  b = 3
  len_ = 100
  # plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

  con1 = sine_wave(len_, a+10, 1)
  con2 = sine_wave(len_, a+4, 4)
  con3 = sine_wave(len_, a+2, 6)
  con4 = sine_wave(len_, a+1, 1)
  con5 = sine_wave(len_, a+9, 8)
  con6 = sine_wave(len_, a+23, 3)
  con7 = sine_wave(len_, a+3, 7)
  con8 = sine_wave(len_, a+2, 2)
  con9 = sine_wave(len_, a+87, 4)
  con10 = sine_wave(len_, a+12, 8)



  sini = np.concatenate([con1, con2, con3, con4, con5, con6, con7, con8, con9, con10])
  return sini


def complex_cx():
  a = 1
  b = 3
  len_ = 100
  #  plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

  cx1 = convex_curve(len_, 1, 2)
  cx2 = convex_curve(len_, 2, 3)
  cx3 = convex_curve(len_, 3, 4)
  cx4 = convex_curve(len_, 4, 5)
  cx5 = convex_curve(len_, 5, 6)
  cx6 = convex_curve(len_, 6, 7)
  cx7 = convex_curve(len_, 7, 8)
  cx8 = convex_curve(len_, 1, 9)
  cx9 = convex_curve(len_, 10, 10)
  cx10 = convex_curve(len_, 5, 1)



  complex_x = np.concatenate([cx1, cx2, cx3, cx4, cx5, cx6, cx7, cx8, cx9, cx10])
  return complex_x

def trup():
  a = 1
  b = 3
  len_ = 100
  #  plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

  con1 = trend_up(len_, 1, 2)
  con2 = trend_up(len_, 2, 3)
  con3 = trend_up(len_, 3, 4)
  con4 = trend_up(len_, 4, 5)
  con5 = trend_up(len_, 5, 6)
  con6 = trend_up(len_, 6, 7)
  con7 = trend_up(len_, 7, 8)
  con8 = trend_up(len_, 1, 9)
  con9 = trend_up(len_, 10, 10)
  con10 = trend_up(len_, 5, 1)



  t_rup = np.concatenate([con1, con2, con3, con4, con5, con6, con7, con8, con9, con10])
  return t_rup

def trdn():
  a = 1
  b = 3
  len_ = 100
  #  plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

  con1 = trend_down(len_, 1, 2)
  con2 = trend_down(len_, 2, 3)
  con3 = trend_down(len_, 3, 4)
  con4 = trend_down(len_, 4, 5)
  con5 = trend_down(len_, 5, 6)
  con6 = trend_down(len_, 6, 7)
  con7 = trend_down(len_, 7, 8)
  con8 = trend_down(len_, 1, 9)
  con9 = trend_down(len_, 10, 10)
  con10 = trend_down(len_, 5, 1)



  t_rdn = np.concatenate([con1, con2, con3, con4, con5, con6, con7, con8, con9, con10])
  return t_rdn

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def const_ds():
  a = 1
  b = 3
  len_ = 100
  #  plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

  con1 = constant(len_, 50+30)
  con2 = constant(len_, 30)
  con3 = constant(len_, 50)
  con4 = constant(len_, 100)
  con5 = constant(len_, 30+30)
  con6 = constant(len_, 80+30)
  con7 = constant(len_, 10+30)
  con8 = constant(len_, 10+10)
  con9 = constant(len_, 20+20)
  con10 = constant(len_, 20+30)

  const_s = np.concatenate([con1, con2, con3, con4, con5, con6, con7, con8, con9, con10])
  return const_s

def complex_concave():
  a = 1
  b = 3
  len_ = 100
  #  plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

  con1 = concave_curve(len_, 1, 2)
  con2 = concave_curve(len_, 2, 3)
  con3 = concave_curve(len_, 3, 4)
  con4 = concave_curve(len_, 4, 5)
  con5 = concave_curve(len_, 5, 6)
  con6 = concave_curve(len_, 6, 7)
  con7 = concave_curve(len_, 7, 8)
  con8 = concave_curve(len_, 1, 9)
  con9 = concave_curve(len_, 10, 10)
  con10 = concave_curve(len_, 5, 1)

  complex_oncave = np.concatenate([con1, con2, con3, con4, con5, con6, con7, con8, con9, con10])
  return complex_oncave


def make_selected_misc_model(models, data_, window, sel=None, epoch=100, misc=False):
  X, y = split_sequence(data_, window)
  # trainX = np.reshape(X, (X.shape[0], 1, X.shape[1]))
  # testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
  model = Sequential()
  # model.add(LSTM(4, input_shape=(1, look_back)))
  # model.add(Dense(2))
  # model.add(Dense(1, input_dim=window, activation='relu'))
  model.add(Dense(5, activation='relu', input_shape=(window,)))
  op = tf.keras.optimizers.Adam(
      learning_rate=0.001,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False)

  X, y = split_sequence(data_, window)
  trainX = np.reshape(X, (X.shape[0], 1, X.shape[1]))
  input1 = keras.Input(shape=(window,))

  # l1 = Dense(1 , activation='relu') # , input_shape=(window,)),

  op = tf.keras.optimizers.Adam(
      learning_rate=0.001,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False
      )
  l = []
  if sel == None :
    for k, v in models.items():
      v.trainable = False
      st = v(input1)
      l.append(st)
  
  else :
    for i in sel:
      models[i].trainable = False
      st = models[i](input1)
      l.append(st)

  if misc :
    misc_layer = Dense(1 , activation='relu')(input1)# , input_shape=(window,)),
    l.append(misc_layer)


  concat_ = keras.layers.Concatenate()(l)
  output = Dense(1)(concat_)
  model = Model(inputs=input1, outputs=output)
  model.compile(loss='mean_squared_error', optimizer=op)
  model.build(input_shape=(window,))
  # model.fit(trainX, y, epochs=epoch, batch_size=1, verbose=0)

  earlyStopping = EarlyStopping(monitor='mean_squared_error', patience=10, verbose=0, mode='min')
  # mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='mean_squared_error', mode='min')

  model.fit(trainX, y, batch_size=1, epochs=epoch, verbose=0, callbacks=[earlyStopping])


  return model


def make_big_other_model(dataset, window, epoch=100, ver=0):
  trainX, trainY = split_sequence(dataset, window)
  trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  model = Sequential()

  model.add(LSTM(units=50, return_sequences=True)) #, input_shape=(trainX.shape[1], 1)))
  model.add(Dropout(0.2))
  model.add(LSTM(units=50, return_sequences=True))
  model.add(Dropout(0.2))

  model.add(LSTM(units=50, return_sequences=True))
  model.add(Dropout(0.2))

  model.add(LSTM(units=50))
  model.add(Dropout(0.2))
  model.add(Dense(units = 1))
  model.compile(optimizer = 'adam', loss = 'mean_squared_error')
  #     model.fit(features_set, labels, epochs = 100, batch_size = 32)
  # model.fit(trainX, trainY, epochs=epoch, batch_size=1, verbose=0)


  earlyStopping = EarlyStopping(monitor='mean_squared_error', patience=10, verbose=ver, mode='min')
  # mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='mean_squared_error', mode='min')

  model.fit(trainX, trainY, batch_size=1, epochs=epoch, verbose=ver, callbacks=[earlyStopping])


  return model

def make_big_model(data_, window, epoch=100):
  batch_size_ = 1
  X, y = split_sequence(data_, window)
  # X = scaler.fit_transform(X)
  trainX = np.reshape(X, (X.shape[0], 1, X.shape[1]))
  # scaler = MinMaxScaler(feature_range=(0, 1))

  model = Sequential()
  model.add(LSTM(5, stateful=True))
  model.add(Dense(1))
  op = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False)

  model.compile(loss='mean_squared_error', optimizer=op)
  model.fit(trainX, y, epochs=epoch, batch_size=batch_size_, verbose=0)
  return model

def make_linear_model(data_, window, epoch=100):
  X, y = split_sequence(data_, window)
  trainX = np.reshape(X, (X.shape[0], 1, X.shape[1]))
  model = Sequential(
    [
      Dense(1 , activation='relu') # , input_shape=(window,)),
    ]
  )
  op = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False)

  model.compile(loss='mean_squared_error', optimizer=op)
  # model.fit(trainX, y, epochs=epoch, batch_size=1, verbose=0)

  earlyStopping = EarlyStopping(monitor='mean_squared_error', patience=10, verbose=0, mode='min')
  # mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='mean_squared_error', mode='min')

  model.fit(trainX, y, batch_size=1, epochs=epoch, verbose=0, callbacks=[earlyStopping])

  return model


def plot_prediction(model, testX, y, cut=-1, plot=True ,st='-'):

  startTime = time.time()
  y_pred = model.predict(testX)  
  executionTime = (time.time() - startTime)

  pre = y_pred[:,0]
  r2 = metrics.r2_score(y, pre)
  mse = metrics.mean_squared_error(y, pre)

  print(f'r2: {r2}, mse: {mse}, cost: {executionTime}')
  if plot:
    plt.figure(num=None, figsize=(8, 6), dpi=160, facecolor='w', edgecolor='k')
    plt.plot(y[:cut],st)
    plt.plot(pre[:cut],'.')
    plt.legend(['real', 'pred'])

def train_(window = 5):
  values ={
    'concave' : complex_concave(),
    'convex' : complex_cx(),
    'sin' : sin(),
    'const' : const_ds(),
    'trend_up' : trup(),
    'trend_down' : trdn(),
    # 'complex' : complex_,
  }

  random.seed(10)
  models = {}
  for k, v in values.items():
    startTime = time.time()
    models[k] = make_linear_model(v, window, 100)
    executionTime = (time.time() - startTime)
    print(f'{k}: {executionTime}')

  co = complex_()
  the_chosen = list(values.keys())
  
  startTime = time.time()
  models['big'] = make_big_other_model(co, window, epoch=100)
  executionTime = (time.time() - startTime)
  print(f'big: {executionTime}')

  startTime = time.time()
  models['all_model'] = make_selected_misc_model(models, co, window, sel=the_chosen, epoch=100, misc=False)
  executionTime = (time.time() - startTime)
  print(f'all_model: {executionTime}')



  startTime = time.time()
  models['all_misc'] = make_selected_misc_model(models, co, window, sel=the_chosen, epoch=100, misc=True)
  executionTime = (time.time() - startTime)
  print(f'all_misc: {executionTime}')

  #  window = 5
  sele = ['convex', 'trend_down', 'concave']
  startTime = time.time()
  models['sel_model'] = make_selected_misc_model(models, co, window, sel=sele, epoch=100, misc=False)
  executionTime = (time.time() - startTime)
  print(f'sel_model: {executionTime}')
  
  #  window = 5

  sele = ['convex', 'trend_down', 'concave']
  startTime = time.time()
  models['sel_misc'] = make_selected_misc_model(models, co, window, sel=sele, epoch=100, misc=True)
  executionTime = (time.time() - startTime)
  print(f'sel_misc: {executionTime}')

  al = list(models.keys())
  print(al)
  return models 

def train_dataset(co, window=5, name='unnamed', ver = 0):
  model = make_big_other_model(co, window, epoch=500, ver=ver)
  model.save(f'../models/{name}-{window}-{int(time.time())}')
  return model


def inference_(models=None, data_path = None, col = 'tps_', window=5, lim=-1, plot_=False, terval=5):
  print(col)
  if models == None:
    models = {}
    models['all'] = load_model('../models/all-model.model')
    models['all_misc'] = load_model('../models/all-misc.model')
    models['sel'] = load_model('../models/optimized.model')
    models['sel_misc'] = load_model('../models/optimized+misc.model')
    models['big'] = load_model('../models/50-50-LSTM.model')
#  Add more as needed

  else :
    pass

  if data_path == None:
    ssd_ = pd.read_csv('../../ssd_cleaned_headers.csv', delimiter=';')
    ssd_tps_ = ssd_[col].values[:lim]
  else:
    ssd_ = pd.read_csv(data_path, delimiter=';')
    ssd_tps_ = ssd_[col].values[:lim]

  data_ = ssd_tps_
  X_test, y_test = split_sequence(data_, window)
  testX = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

  for k, v in models.items():
    print(k)
    for i in range(terval):
      plot_prediction(v, testX, y_test, plot = plot_)
      plot_prediction(v, testX, y_test, plot = False)
      
      
def main_():
  cols = ['tps_', 'rd_sec/s_', 'wr_sec/s_', 'Avgrq-sz_']
  data_paths = ['../datasets/nvme.csv']
  win_max = 10
  models = {}
  w = 5
  #  for w in range(6, 10):
  print(f'Window : {w}')
  #  #  models =  train_(window=w)
  #  for i in cols:
  #    for j in data_paths:
  #      #  inference_(None, j, col=i)
  #      inference_(None, j, window=w, lim=6)

  print(f"{'*'*20} ONLY big models now {'*'*20}")

  #  for w in range(5, win_max):
  for i in data_paths:
    for j in cols:
      print(f'{w} {i} {j}')
      le_csv = pd.read_csv(i, delimiter=';')
      le_vals, le_test_vals = le_csv[j].values[:10000], le_csv[j].values[10000:15000]
      models[j] = train_dataset(le_vals, w, f'nvme-{j}', ver = 0)

if __name__ == '__main__':
  main_()
