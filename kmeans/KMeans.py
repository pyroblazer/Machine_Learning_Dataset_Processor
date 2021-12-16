import numpy as np
import math
import copy as copy

class KMeans():
	def __init__(self, n_clusters=3, max_iter=100):
		self.n_clusters = n_clusters
		self.max_iter = max_iter

	# Returns an array of positions
	def initialize_clusters(self):
		self.init_indexes = np.random.randint(low= 0, high=len(self.data), size=self.n_clusters)
		# For each index in random_indexes, get the data
		clusters = []
		self.cluster_centers_ = []
		self.components = self.data.iloc[0].keys()
		for idx in self.init_indexes:
			# Append selected row to the clusters
			self.cluster_centers_.append(self.data.iloc[idx])
			cluster = []
			clusters.append(cluster)
		return clusters
	
	def recomputeMidpoint(self):
		for i in range(len(self.cluster_centers_)):
			newMidpoint = dict()
			for component in self.components:
				newMidpoint[component] = 0
			for point in self.clusters[i]:
				for component in self.components:
					newMidpoint[component] += point[component]
			for component in self.components:
				if (len(self.clusters[i]) != 0):
					newMidpoint[component] /= len(self.clusters[i])
			self.cluster_centers_[i] = newMidpoint
		self.clusters = []	
		for i in range(self.n_clusters):
			cluster = []
			self.clusters.append(cluster)

	def getDistance(self, point, midpoint):
		sum = 0
		for component in self.components:
			sum += (point[component] - midpoint[component])**2
		return math.sqrt(sum)

	def getNearestCluster(self, point):
		min = np.inf
		retCluster = None
		for i in range(len(self.cluster_centers_)):
			distance = self.getDistance(point, self.cluster_centers_[i])
			if (distance < min):
				min = distance
				retCluster = self.clusters[i]
		return retCluster

	def getNearestClusterIdx(self, point):
		min = np.inf
		retClusterIdx = None
		for i in range(len(self.cluster_centers_)):
			distance = self.getDistance(point, self.cluster_centers_[i])
			if (distance < min):
				min = distance
				retClusterIdx = i
		return retClusterIdx

	def assignCluster(self, point):
		cluster = self.getNearestCluster(point)
		cluster.append(point)
	
	def fit(self, data):
		self.data = data
		self.clusters = self.initialize_clusters()
		self.prev_clusters = []

		for iter in range(self.max_iter):
			for i in range(len(self.data)):
				self.assignCluster(self.data.iloc[i])
			
			if (iter != 0):
				isConvergent = True
				for idx_cluster in range(self.n_clusters):
					for component in self.components:
						if (self.cluster_centers_[idx_cluster][component] != self.prev_clusters[idx_cluster][component]):
							isConvergent = False
							break
					if (isConvergent == False):
						break
				if (isConvergent):
					break

			self.prev_clusters = copy.copy(self.cluster_centers_)
			self.recomputeMidpoint()

	# Called after fit
	def predict(self, data):
		self.labels_ = []
		for i in range(len(data)):
			point = data.iloc[i]
			self.labels_.append(self.getNearestClusterIdx(point))
		return self.labels_

	def print(self):
		for cluster in self.clusters:
			print(len(cluster))
		for midpoint in self.cluster_centers_:
			for component in self.components:
				print(midpoint[component], end=",  ")
			print()