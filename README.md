# KMeans ++
This is an Implementation of the KMeans ++ algorithim. the included file KMeans.py includes a class that clusters the data then plots the resulting clusters.


```python
import numpy as np
import matplotlib.pylab as plt
from KMeans import KMeans
```


```python
from sklearn.datasets import make_blobs
X,y = make_blobs(centers=4, n_samples=500, random_state=0, cluster_std=0.7)
class_ = KMeans(X, 4, iterations = 500)
class_.train()
```


![png](output_2_0.png)



```python
data=np.loadtxt("s1.txt")
class_ = KMeans(data, 15, iterations = 1000)
class_.train()
```


![png](output_3_0.png)



```python
data=np.loadtxt("s3.txt")
class_ = KMeans(data, 15, iterations = 500)
class_.train()
```


![png](output_4_0.png)

