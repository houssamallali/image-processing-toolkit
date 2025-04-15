import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

# Fonction de génération de points aléatoires autour d’un centre comme x,y
def generation(n, x, y):
    Y = np.random.randn(n, 2) + np.array([[x, y]])
    return Y

# Génération des 3 groupes de points
points1 = generation(100, 0, 0)
points2 = generation(100, 3, 4)
points3 = generation(100, -5, -3)

# Concaténation des groupes
pts = np.concatenate((points1, points2, points3))

# Affichage des points avant classification
plt.plot(pts[:, 0], pts[:, 1], 'ro')
plt.title("Nuage de points générés aléatoire")
plt.show()

''' ça c'est le nombre de clusters'''
n = 3

# Initialisation k-means avec n_init=10
k_means = KMeans(init="k-means++", n_clusters=n, n_init=10)
t0 = time.time()
k_means.fit(pts)
t_batch = time.time() - t0

k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

# Affichage des résultats
fig = plt.figure()
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

for k, col in zip(range(n), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(pts[my_members, 0], pts[my_members, 1], 'w',
             markerfacecolor=col, marker='.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('KMeans')
plt.show()
