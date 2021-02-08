# shape-stream
Algorithms for streaming clustering of time series events. 

This package contains algorithms which identify and update k time series event clusters. The clusters are updated as new events are added to the dataset in a streaming fashion. The underlying clustering approach builds on kShape [1], which uses normalized cross correlations as a distance metric. 
So far, the kshape.py file contains two streaming time series clustering algorithms:.
1. kShapeStream - A simple streaming version of kShape
2. kShapeProbStream - A modification of kShapeStream which assigns points to a cluster based on a probablistic distance

For a quick demo of how to use the algorithms in this package, take a look at the Clustering Demo Jupyter notebook. For questions, comments, or bugs, contact mohini@berkeley.edu

**References**

[1] Paparrizos, John, and Luis Gravano. "k-shape: Efficient and accurate clustering of time series." Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data. 2015.
