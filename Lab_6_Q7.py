import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, metrics
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Data/Country-data.csv")#, header=None)

x = df.drop(columns="country")
print(df)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import completeness_score
kma=cluster.KMeans(n_clusters=3)
#scaled = StandardScaler()
#x = scaled.fit_transform(df)


kma.fit(x)
print(kma.labels_)

plt.scatter(df["income"], df["life_expec"], c = kma.labels_)
plt.show()

df["class"] = kma.labels_

print(df)

classification = df.groupby("class").apply(lambda x: x["country"].unique())

print(classification)

## class 0, underdeveloped
## class 1, developeing
## class 2, developed