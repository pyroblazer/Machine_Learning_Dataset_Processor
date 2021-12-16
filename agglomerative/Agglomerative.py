import numpy as np
import math
import copy as copy

class Agglomerative():
	def __init__(self, n_clusters=3, linkage="average-group"):
		self.n_clusters = n_clusters
		self.linkage = linkage
		self.clusters = []
		self.inactivated = []

	def initDistanceMatrix(self):
		self.distance = []
		for i in range(len(self.clusters)):
			self.distance.append([])
			for j in range(len(self.clusters)):
				self.distance[i].append([])
		for i in range(len(self.clusters)):
			for j in range(len(self.clusters)):
				dist = 0
				for component in self.components:
					dist += (self.clusters[i][0][component] - self.clusters[j][0][component])**2
				self.distance[i][j] = math.sqrt(dist)
				if (i > j):
					self.distance[i][j] = -1

	def getClusterIndexPair(self):
		source = None
		target = None
		minDistance = np.inf
		for i in range(len(self.distance)):
			for j in range(len(self.distance)):
				distance = self.getClusterDistance(i, j)
				if (distance<minDistance):
					source = i
					target = j
					minDistance = distance
		return source, target

	def getClusterDistance(self, sourceIdx, targetIdx):
		if (sourceIdx in self.inactivated or targetIdx in self.inactivated or sourceIdx == targetIdx):
			return np.inf
		if (sourceIdx > targetIdx):
			return self.getClusterDistance(targetIdx, sourceIdx)
		return self.distance[sourceIdx][targetIdx]

	def putClusterDistances(self, sourceIdx, targetIdx, value):
		if (sourceIdx in self.inactivated):
			raise Exception("404 Source not found")
		self.distance[sourceIdx][targetIdx] = value
		self.distance[targetIdx][sourceIdx] = value

	def updateClusters(self):
		source, target = self.getClusterIndexPair()
		print(source, target)
		for i in range(len(self.clusters)):
			self.putClusterDistances(source, i, min(self.getClusterDistance(source, i), self.getClusterDistance(target, i)))
		self.inactivated.append(target)
		targetCluster = self.clusters[target]
		for data in targetCluster:
			self.clusters[source].append(data)

	def fit(self, X):
		self.components = X.iloc[0].keys()
		self.clusters = []
		for i in range(len(X)):
			data = dict()
			for component in self.components:
				data[component] = X.iloc[i][component]
			cluster = []
			cluster.append(data)
			self.clusters.append(cluster)
		self.initDistanceMatrix()
		while(len(self.clusters) - len(self.inactivated) > self.n_clusters):
			print(len(self.clusters) - len(self.inactivated))
			self.updateClusters()
		for i in range(len(self.clusters)):
			if i not in self.inactivated:
				print(i)
				data = dict()
				for component in self.components:
					data[component] = 0
				for point in self.clusters[i]:
					for component in self.components:
						data[component] += point[component]
				for component in self.components:
					data[component] /= len(self.clusters[i])
				print(data)

	