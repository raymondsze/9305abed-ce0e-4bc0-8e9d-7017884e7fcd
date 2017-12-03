# LSTM for international airline passengers problem with regression framing
import numpy
import json
import locale
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer='adam')

last12Data = numpy.array([dataset[len(dataset) -12:]])
lastX = numpy.reshape(last12Data, (last12Data.shape[0], last12Data.shape[1], 2))

newPredict = []

# print testX
for i in range(36):
	predict = model.predict(lastX, batch_size=batch_size)
	month = 0.09090906 + lastX[0][len(lastX[0])-1][1]
	while month > 1:
		month = month-1
	print(month)
	sold_units = predict[0][0]
	newPredict.append([sold_units])
	lastX[0] = numpy.concatenate((lastX[0], [[sold_units, month]]), axis=0)[1:]
	# model.reset_states()

newPredict = numpy.array(newPredict)
print(newPredict)

newPredict = scaler.inverse_transform(list(map(lambda x:[x, 0], newPredict)))

newPredictPlot = numpy.empty_like(newPredict)
newPredictPlot[:, :] = numpy.nan
newPredictPlot[:, :] = newPredict

# plot baseline and predictions
plt.plot(list(map(lambda x:x[0], newPredictPlot)))
plt.show()