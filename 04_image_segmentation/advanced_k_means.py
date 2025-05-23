from mpl_toolkits.mplot3d import Axes3D
import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# ===========================
# Chargement de l'image couleur
# ===========================
cells = imageio.imread('../images/Tv16.png')
nLines, nCols, channels = cells.shape

# ===========================
# Reshape des données en (N, 3)
# ===========================
data = np.reshape(cells, (nLines * nCols, channels))
data = data.astype(np.float32)

# ===========================
# Application du KMeans
# ===========================
k_means = KMeans(n_clusters=3, n_init=10)
k_means.fit(data)

# ===========================
# Conversion du résultat en image
# ===========================
segmentation = 70 * np.reshape(k_means.labels_, (nLines, nCols))

# ===========================
# Affichage du résultat
# ===========================
fig = plt.figure()
plt.imshow(segmentation, cmap=plt.cm.gray)
plt.title("Résultat de la segmentation")
plt.axis("off")
plt.show()

# ===========================
# Sauvegarde de l'image
# ===========================
imageio.imwrite("segmentation_kmeans.png", segmentation.astype(np.uint8))

# ===========================
# Affichage 3D des clusters RGB
# ===========================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

for k, col in zip(range(3), colors):
    my_members = k_means.labels_ == k
    cluster_center = k_means.cluster_centers_[k]
    ax.scatter(
        data[my_members, 0],
        data[my_members, 1],
        data[my_members, 2],
        c=col,
        s=10,
        alpha=0.6,
        edgecolors='k',
        linewidths=0.2
    )
    ax.scatter(
        cluster_center[0],
        cluster_center[1],
        cluster_center[2],
        c='black',
        s=250,
        edgecolors='white',
        marker='*',
        linewidths=1.5,
        label=f'Center {k + 1}'
    )

ax.set_xlabel("Red")
ax.set_ylabel("Green")
ax.set_zlabel("Blue")
ax.legend()
ax.grid(True)

plt.title("3D RGB clustering")
plt.show()