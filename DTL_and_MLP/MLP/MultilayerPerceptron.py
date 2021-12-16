import pandas as pd
import numpy as np
import math
# from sklearn.neural_network import MLPClassifier # neural network
# from sklearn.model_selection import train_test_split
# from sklearn import metrics

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

class Perceptron:

	def __init__(self, rate, input_length):
		self.data = []
		self.weight = []
		self.delta_weight = []
		self.rate = rate

		random_matrix = np.random.randn(1, input_length) * np.sqrt(1 / input_length)
		for rand_array in random_matrix:
			for rand_num in rand_array:
				self.weight.append(rand_num)
		
		# print(self.weight)

		for inp in range(input_length):
			self.delta_weight.append(0)
			

	def input_data(self, data):
		self.data = []
		for datum in data:
			self.data.append(datum)

	def calc_sigmoid(self):
		jumlah = 0
		for i in range(len(self.data)):
			jumlah += self.data[i] * self.weight[i]
		self.output = sigmoid(jumlah)

	#for backprop
	def calc_delta(self, multiplier):
		self.delta = self.output * (1-self.output) * multiplier

	def update_delta_weight(self):
		for i in range(len(self.delta_weight)):
			self.delta_weight[i] += self.rate * self.delta * self.data[i]
	
	# End of batch-size
	def update_weight(self):
		for i in range(len(self.weight)):
			self.weight[i] += self.delta_weight[i]
			self.delta_weight[i] = 0
	
	def load_weight(self, weights_from_file):
		for i in range(len(weights_from_file)):
			self.weight[i] = weights_from_file[i]

class myMLP:

	def __init__(self, hidden_layer_sizes=[10, 5, 2], learning_rate=0.05, max_iter=400, error_treshold=0.0001, batch_size=32):
		# Attributes
		self.layers = []
		self.hidden_layer_sizes = hidden_layer_sizes
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.error_treshold = error_treshold
		self.batch_size = batch_size
		self.output_print = []
		# Model output file
		self.output_file = []
		self.is_loaded_from_file = False


	def fit(self, data_inputs, target):
		self.data_inputs = data_inputs
		self.target = target
		self.classes = self.target.unique()

		if (not self.is_loaded_from_file):
			try:
				number_of_inputs_from_previous_layer = len(self.data_inputs.columns)
				# Initialize perceptrons in the hidden layers (from index 1)
				for layer_idx in range(len(self.hidden_layer_sizes)):
					# hidden_layer = Array of perceptrons
					number_of_perceptrons_current_layer = self.hidden_layer_sizes[layer_idx]
					hidden_layer = self.initialize_perceptrons_in_layer(number_of_perceptrons_current_layer, number_of_inputs_from_previous_layer)
					number_of_inputs_from_previous_layer = self.hidden_layer_sizes[layer_idx]
					self.layers.append(hidden_layer)

				# Construct last (output) layer of perceptrons
				number_of_perceptrons_last_layer = len(self.target.unique())
				number_of_inputs_from_previous_layer = self.hidden_layer_sizes[-1]

				output_layer = self.initialize_perceptrons_in_layer(number_of_perceptrons_last_layer, number_of_inputs_from_previous_layer)
				self.layers.append(output_layer)

			except Exception as e:
				print(e)
				# Construct last (output) layer of perceptrons
				number_of_perceptrons_last_layer = len(self.target.unique())
				number_of_inputs_from_previous_layer = len(self.data_inputs.columns)
				output_layer = self.initialize_perceptrons_in_layer(number_of_perceptrons_last_layer, number_of_inputs_from_previous_layer)
				self.layers.append(output_layer)

		# Start feed forward and backward prop
		number_of_rows = len(data_inputs)
		for iteration in range(self.max_iter):
			error_total = 0
			for row in range(number_of_rows):
				# print("row")
				# print(row)
				self.feed_forward(row)

				# Do backward prop then get error
				error = self.backward_prop(row)
				error_total += error
				
				if (row % self.batch_size == 0):
					self.update_all_weights()

			self.update_all_weights()

			if (error_total < self.error_treshold):
				break

	def update_all_weights(self):
		for layer in self.layers:
			for perceptron in layer:
				perceptron.update_weight()

	def calculate_error(self, diff):
		return 0.5 * (diff ** 2)

	def initialize_perceptrons_in_layer (self, number_of_perceptrons, number_of_inputs):
		layer = []
		for idx_perceptron in range(number_of_perceptrons):
			layer.append(Perceptron(self.learning_rate, number_of_inputs+1))
		return layer

	def feed_forward(self, row):
		inputs = []
		outputs = []
		# Initial inputs
		for column in self.data_inputs.columns:
			inputs.append(self.data_inputs[column][row])
		inputs.append(1)

		for layer_idx in range(len(self.layers)):
			outputs.clear()
			for perceptron in self.layers[layer_idx]:
				perceptron.input_data(inputs)
				perceptron.calc_sigmoid()
				outputs.append(perceptron.output)

			inputs.clear()
			for output_data in outputs:
				inputs.append(output_data)

			inputs.append(1)

	def backward_prop(self, row):
		# Last layer
		total_error = 0
		for i in range(len(self.layers[-1])):
			perceptron = self.layers[-1][i]
			# Calculate diff (multiplier):
			if self.classes[i] == self.target[row]:
				result = 1
			else:
				result = 0
			diff = result - perceptron.output
			perceptron.calc_delta(diff)
			perceptron.update_delta_weight()
			total_error += self.calculate_error(diff)

		# Hidden layers
		for layer_idx in range(len(self.layers)-1): #banyaknya layer di layers, kecuali output layer
			layer_size = len(self.layers[-layer_idx-2]) #banyaknya perceptron di layer itu
			for perc_idx in range(layer_size): #untuk setiap perceptron di layer itu
				diff = 0
				for next_perceptron in self.layers[-layer_idx-1]:

					diff += next_perceptron.delta * next_perceptron.weight[perc_idx]
				self.layers[-layer_idx-2][perc_idx].calc_delta(diff)
				self.layers[-layer_idx-2][perc_idx].update_delta_weight()

		return total_error

	def predict(self, data_inputs):
		inputs = []
		outputs = []
		predictions = []
		for row in range(len(data_inputs)):
			inputs.clear()
			outputs.clear()
			# Initial inputs
			for column in data_inputs.columns:
				inputs.append(data_inputs[column][row])
			inputs.append(1)

			for layer_idx in range(len(self.layers)):
				outputs.clear()
				for perceptron in self.layers[layer_idx]:
					perceptron.input_data(inputs)
					perceptron.calc_sigmoid()
					outputs.append(perceptron.output)
				inputs.clear()
				for output in outputs:
					inputs.append(output)
				inputs.append(1)
			idx = outputs.index(max(outputs))
			predictions.append(self.classes[idx])
		return predictions

	def save_model(self):
		self.output_print.clear()
		self.output_file.clear()
		for layer_idx in range(len(self.layers)):
			for perceptron_idx in range(len(self.layers[layer_idx])):
				for weight_idx in range(len(self.layers[layer_idx][perceptron_idx].weight)):
					self.output_file.append(str(self.layers[layer_idx][perceptron_idx].weight[weight_idx]))
					if (weight_idx != len(self.layers[layer_idx][perceptron_idx].weight) - 1):
						self.output_print.append(str("Weight " + str(weight_idx) + "-" + "[" + str(layer_idx) + "][" + str(perceptron_idx) + "]: " + str(self.layers[layer_idx][perceptron_idx].weight[weight_idx])))
					else:
						self.output_print.append(str("Bias " + "[" + str(layer_idx) + "][" + str(perceptron_idx) + "]: " + str(self.layers[layer_idx][perceptron_idx].weight[weight_idx])))
				self.output_file.append("p")
			# New layer
			self.output_file.append("l")


	def show_model(self, n=None):
		self.save_model()
		if (n is None):
			for output in self.output_print:
				print(output)
		else:
			for i in range(n):
				print(self.output_print[i])

	def save_model_to_file(self, filename, n=None):
		self.save_model()
		f = open(filename, "w+")
		for i in range(len(self.output_file)):
			# New line every "l"
			if (self.output_file[i] == "l"):
				f.write("\n")
			else:
				f.write(self.output_file[i] + " ")
		f.close()
	
	def load_perceptron_from_file(self, number_of_weights_include_bias, weights_from_file_per_perceptron):
		perceptron = Perceptron(self.learning_rate, number_of_weights_include_bias)
		perceptron.load_weight(weights_from_file_per_perceptron)
		return perceptron

	def load_model_from_file(self, filename, n=None):
		f = open(filename, "r")
		# If file is in open mode, then proceed
		if (f.mode == 'r'):
			self.layers.clear()
			# Read per line
			lines = f.readlines()
			for line in lines:
				# Split by " "
				values = line.split()
				weights_from_file_per_perceptron = []
				one_layer_of_perceptrons = []
				input_count = 0
				for value in values:
					if (value == 'p'):
						perceptron = self.load_perceptron_from_file(input_count, weights_from_file_per_perceptron)
						one_layer_of_perceptrons.append(perceptron)
						input_count = 0
						weights_from_file_per_perceptron.clear()
					else:
						weights_from_file_per_perceptron.append(float(value))
						input_count = input_count + 1
				self.layers.append(one_layer_of_perceptrons)
		self.is_loaded_from_file = True
		# self.show_model()
		f.close()
