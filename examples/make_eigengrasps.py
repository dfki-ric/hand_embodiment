import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


X = np.loadtxt("relative_markers.csv")
df = pd.DataFrame(X)
df = df.dropna()
X_filtered = df.to_numpy()
pca = PCA(n_components=3)
pca.fit(X_filtered)
X_pca = pca.transform(X_filtered)
sample = pca.inverse_transform(X_pca[0])
np.savetxt("sample3.txt", sample)
