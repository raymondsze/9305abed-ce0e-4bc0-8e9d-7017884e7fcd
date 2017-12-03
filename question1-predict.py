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

filename = 'godiva.json'
json_data = json.load(open(filename))

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# int(data['emerging_usd'].replace(",", ""))
data = numpy.array(list(map(lambda data: [int(data['emerging_usd'].replace(",", "")), months.index(data['month'])], json_data['result']['records'])))

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pd.DataFrame(data)

dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

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

batch_size = 1

months_map = [0, 0.09090909, 0.18181819, 0.27272728, 0.36363637, 0.45454547, 0.54545456, 0.63636363, 0.72727275, 0.81818187, 0.90909094, 1]
# print testX
for i in range(36):
	predict = loaded_model.predict(lastX, batch_size=batch_size)
	month = months_map[(i+10)%12]
	# print(month)
	sold_units = predict[0][0]
	newPredict.append([sold_units])
	lastX[0] = numpy.concatenate((lastX[0], [[sold_units, month]]), axis=0)[1:]
	# model.reset_states()

newPredict = numpy.array(newPredict)
print(newPredict)

newPredict = scaler.inverse_transform(list(map(lambda x:[x, 0], newPredict)))

newPredictPlot = numpy.empty_like(newPredict)
newPredictPlot.resize(len(dataset)+len(newPredict), 2)
newPredictPlot[:, :] = numpy.nan
newPredictPlot[len(dataset):, :] = newPredict

# plot baseline and predictions
plt.plot(list(map(lambda x:x[0], scaler.inverse_transform(dataset))))
plt.plot(list(map(lambda x:x[0], newPredictPlot)))
plt.show()