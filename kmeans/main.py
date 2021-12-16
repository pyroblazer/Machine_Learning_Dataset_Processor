import pandas as pd
import copy
import numpy as np
from KMeans import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.cluster import fowlkes_mallows_score

data = pd.read_csv("iris.csv")
target = data.species
data_copy = copy.copy(data)

# Drop the class
inputs = data.drop('species', axis = 1)

# Test from n_clusters = 2 until n_clusters = 6
for n_clusters in range(2, 6+1):
    # Fowkes-Mallows and Silhouette evaluation:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(inputs)
    labels = np.array(kmeans.predict(inputs))
    print("labels")
    print(labels)

    print("n_clusters =", n_clusters)

    print("Menggunakan metode Fowlkes-Mallows: ")
    fowlkes_mallows = fowlkes_mallows_score(labels, target)
    print("Fowlkes Mallows Score:", fowlkes_mallows)

    print("Menggunakan metode Silhouette:")
    silhouette_avg = silhouette_score(inputs, labels)
    print("Hasil rata-rata skor silhouette:", silhouette_avg)
    print()
    print()

    silhouette_values_per_point = silhouette_samples(inputs, labels)

    # Visualize Silhouette subplot
    # 1 row and 2 columns: Left -> silhouette plot and Right -> Cluster Visualization
    fig, silhouette_viz = plt.subplots(1)
    fig.set_size_inches(18, 7)

    # Silhouette score ranges from -1 until 1
    silhouette_viz.set_xlim([-0.2, 1])
    silhouette_viz.set_ylim([0, len(inputs) + (n_clusters + 1) * 10])

    y_lower = 10
    
    for idx_cluster in range(n_clusters):
        cluster_silhouette_values = silhouette_values_per_point[labels == idx_cluster]
        # See the variance
        cluster_silhouette_values.sort()
        y_upper = y_lower + len(cluster_silhouette_values)
        silhouette_viz.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_values, alpha=0.7)
        silhouette_viz.text(-0.15, y_lower + 0.5 * len(cluster_silhouette_values), str(idx_cluster))
        y_lower = y_upper + 10

    silhouette_viz.set_xlabel("Nilai Skor Silhouette")
    silhouette_viz.set_ylabel("Nomor Cluster")

    silhouette_viz.set_yticks([])  # Clear the yaxis labels / ticks
    silhouette_viz.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    silhouette_viz.set_title("Visualisasi Silhouette Plot untuk n = " + str(n_clusters))

    fig2 = plt.figure()
    cluster_viz = fig2.add_subplot(111, projection='3d')

    # Visualization for Clusters
    img = cluster_viz.scatter(data["sepal_length"], data["sepal_width"], data["petal_length"], c=data["petal_width"], cmap=plt.hot())

    cluster_viz.set_xlabel("Sepal length")
    cluster_viz.set_ylabel("Sepal width")
    cluster_viz.set_zlabel("Petal length")

    cluster_viz.set_title("Visualisasi Cluster untuk n = " + str(n_clusters))

    fig2.colorbar(img).ax.set_ylabel("Petal width")

    center = kmeans.cluster_centers_
    # print(kmeans.cluster_centers_)

    # For visualization purposes
    data_viz_dict = {}

    for key in center[0].keys():
        data_viz_dict[key] = []

    for i in range (len(center)):
        for key in center[i].keys():
            data_viz_dict[key].append(center[i][key])

    cluster_viz.scatter(data_viz_dict["sepal_length"], data_viz_dict["sepal_width"], data_viz_dict["petal_length"],
                        c=data_viz_dict["petal_width"], marker="o", edgecolor="k", s= 20*4**3, alpha=1)

    for i in range (len(center)):
        cluster_viz.scatter(center[i]["sepal_length"], center[i]["sepal_width"], center[i]["petal_length"],
                            c="red", marker='$%d$' % i, s=20*4**2, alpha=1)

plt.show()
