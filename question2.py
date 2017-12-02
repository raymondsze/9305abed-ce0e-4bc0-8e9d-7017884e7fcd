import numpy
import json

filename = 'godiva.json'
json_data = json.load(open(filename))
temperature_filename = 'temperature.json'
temperature_data = json.load(open(temperature_filename))
stock_filename = 'stock.json'
stock_data = json.load(open(stock_filename))

data = numpy.array(list(map(lambda (i, data): [int(data['emerging_usd'].replace(",", "")), temperature_data[i] ], enumerate(json_data['result']['records']))))
print(data)

data = numpy.array(list(map(lambda (i, data): [int(data['emerging_usd'].replace(",", "")), stock_data[i] ], enumerate(json_data['result']['records']))))
print(data)