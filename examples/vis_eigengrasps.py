import numpy as np
import pandas as pd
import pytransform3d.visualizer as pv
from sklearn.decomposition import PCA


def animation_callback(step, markers, pca, lo, hi, z):
    z += 0.02 * np.random.randn(pca.n_components)
    z = np.clip(z, lo, hi)
    sample = pca.inverse_transform(z)
    positions = sample.reshape(-1, 3)
    markers.set_data(positions)
    return markers


X = np.loadtxt("relative_markers.csv")
df = pd.DataFrame(X)
df = df.dropna()
X_filtered = df.to_numpy()
pca = PCA(n_components=3)
pca.fit(X_filtered)
Y = pca.transform(X_filtered)
lo = np.min(Y, axis=0)
hi = np.max(Y, axis=0)
z = np.zeros(pca.n_components)

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
markers = fig.scatter(np.zeros((X.shape[1] // 3, 3)), s=0.006)
fig.view_init()
fig.animate(animation_callback, 1, loop=True, fargs=(markers, pca, lo, hi, z))
fig.show()
