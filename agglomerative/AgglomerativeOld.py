import numpy as np
import math
import copy as copy

class AgglomerativeOld(object):
	
	def __init__(self, n_clusters=3, linkage="average-group"):
		self.n_clusters = n_clusters
		self.linkage = linkage
		self.clusters = []
		#added for predict
		self.cluster_centers = []
		self.prediction_cluster_center_for_data = []
		self.prediction_cluster_center_label_for_data = []

	def getPointDistance(self, sourcePoint, targetPoint):
		sum = 0
		for component in self.components:
			sum += (targetPoint[component] - sourcePoint[component]) ** 2
		distance = math.sqrt(sum)
		return distance

	def getSingleDistance(self, sourceCluster, targetCluster):
		minDistance = np.inf
		for i in sourceCluster:
			sourcePoint = self.data.iloc[i]
			for j in targetCluster:
				targetPoint = self.data.iloc[j]
				distance = self.getPointDistance(sourcePoint, targetPoint)
				if (distance < minDistance):
					minDistance = distance
		return minDistance

	def getCompleteDistance(self, sourceCluster, targetCluster):
		maxDistance = -1 * np.inf
		for i in sourceCluster:
			sourcePoint = self.data.iloc[i]
			for j in targetCluster:
				targetPoint = self.data.iloc[j]
				distance = self.getPointDistance(sourcePoint, targetPoint)
				if (distance > maxDistance):
					maxDistance = distance
		return maxDistance

	def getAverageDistance(self, sourceCluster, targetCluster):
		averageDistance = 0
		for i in sourceCluster:
			sourcePoint = self.data.iloc[i]
			averageDistancePerOneObject = 0
			for j in targetCluster:
				targetPoint = self.data.iloc[j]
				distance = self.getPointDistance(sourcePoint, targetPoint)
				averageDistancePerOneObject += distance
			averageDistancePerOneObject /= len(targetCluster)
			averageDistance += averageDistancePerOneObject
		averageDistance /= len(sourceCluster)	
		return averageDistance

	def getClusterCenterPoint(self, cluster):
		center = {}
		for component in self.components:
			center[component] = 0
			for i in cluster:
				center[component] += self.data.iloc[i][component]
			center[component] /= len(cluster)
		return center
		
	def getAverageGroupDistance(self, sourceCluster, targetCluster):
		sourcePoint = self.getClusterCenterPoint(sourceCluster)
		targetPoint = self.getClusterCenterPoint(targetCluster)
		return self.getPointDistance(sourcePoint, targetPoint)

	def getClusterDistance(self, sourceIdx, targetIdx):
		sourceCluster = self.clusters[sourceIdx]
		targetCluster = self.clusters[targetIdx]
		if (self.linkage == "single"):
			return self.getSingleDistance(sourceCluster, targetCluster)
		elif (self.linkage == "complete"):
			return self.getCompleteDistance(sourceCluster, targetCluster)
		elif (self.linkage == "average"):
			return self.getAverageDistance(sourceCluster, targetCluster)
		elif (self.linkage == "average-group"):
			return self.getAverageGroupDistance(sourceCluster, targetCluster)
		else:
			raise Exception("404 Linkage not found")

	def getClusterIndexPair(self):
		source = None
		target = None
		minDistance = np.inf
		for i in range(len(self.clusters)):
			for j in range(len(self.clusters)):
				if (j>i):
					distance = self.getClusterDistance(i, j)
					if (distance<minDistance):
						source = i
						target = j
						minDistance = distance
		return source, target

	def updateClusters(self):
		source, target = self.getClusterIndexPair()
		targetCluster = self.clusters.pop(target)
		for data in targetCluster:
			self.clusters[source].append(data)

	def fit(self, X):
		self.data = X
		self.components = self.data.iloc[0].keys()
		# print(self.components)
		for i in range(len(X)):
			self.clusters.append([i])
		while(len(self.clusters) > self.n_clusters):
			self.updateClusters()
		#self.cluster_centers = []
		for cluster in self.clusters:
			self.cluster_centers.append(self.getClusterCenterPoint(cluster))
		# print(self.cluster_centers)

	def getNearestClusterIdx(self, point):
		min = np.inf
		retClusterIdx = None
		for i in range(len(self.cluster_centers)):
			distance = self.getPointDistance(point, self.cluster_centers[i])
			if (distance < min):
				min = distance
				retClusterIdx = i
		return retClusterIdx
	
	def predict(self, data):
		#classify array of data to respective cluster centers, return cluster label
		self.data = data
		datum_centers = []
		predicted_cluster_label_for_data = []
		data_clusters = []
		for index in range(len(data)):
			datum_centers.append(self.getClusterCenterPoint([index]))
		for datum_center in datum_centers:
			predicted_cluster_label_for_datum = self.getNearestClusterIdx(datum_center)
			predicted_cluster_label_for_data.append(predicted_cluster_label_for_datum)
		return predicted_cluster_label_for_data

	def print(self):
		for cluster in self.clusters:
			print (cluster)
			print(len(cluster))
		for midpoint in self.cluster_centers:
			for component in self.components:
				print(midpoint[component], end=",  ")
			print()


#initialize data
# dfObj = pd.DataFrame(columns=['x','y'])
# dfObj = dfObj.append({'x':1,'y':2}, ignore_index=True)
# dfObj = dfObj.append({'x':1,'y':4}, ignore_index=True)
# dfObj = dfObj.append({'x':1,'y':0}, ignore_index=True)
# dfObj = dfObj.append({'x':4,'y':2}, ignore_index=True)
# dfObj = dfObj.append({'x':4,'y':4}, ignore_index=True)
# dfObj = dfObj.append({'x':4,'y':0}, ignore_index=True)

# print("Dataframe Contents ", dfObj, sep='\n')

# agglo_test = Agglomerative(n_clusters=2, linkage="single")
# agglo_test.fit(dfObj)

# dfTest = pd.DataFrame(columns=['x','y'])
# dfTest = dfTest.append({'x':1,'y':3}, ignore_index=True) #cluster (1,2)
# dfTest = dfTest.append({'x':4,'y':3}, ignore_index=True) #cluster (4,2)
# dfTest = dfTest.append({'x':2.5,'y':2}, ignore_index=True) #middle

# print("Test Contents ", dfTest, sep="\n")

# agglo_test.predict(dfTest)

# agglo_test.print()