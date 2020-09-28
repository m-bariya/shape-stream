# Imports
import math
import numpy as np

from numpy.random import randint
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft

import matplotlib.pyplot as plt

# Iterative kShape
# A kShape object is initialized with a certain k
# The object has k cluster centers and k shape matrices
# 
# Features
# cluster(X) - clusters the data in X using the initial 
# centroids. 

class kShapeStream:
    
    # Initialize
    # Create a kShapeStream object with a certain number
    # of clusters. 
    def __init__(self, k, m):
        # Number of clusters
        self.k = k;
        # Number of times points
        self.m = m; 
        # Centroids : Initially zero
        self.centers = np.zeros([k, m]); 
        # Counts of elements in each cluster
        self.counts = np.zeros([k]); 
        # Shape Matrices from which centroids
        # are extracted. 
        self.S = np.zeros([k, m, m]); 
        # Whether to initialize cluster membership
        self.init_random = True; 
    
    # Time series normalization
    def zscore(self, a, axis=0, ddof=0):
        a = np.asanyarray(a)
        mns = a.mean(axis=axis)
        sstd = a.std(axis=axis, ddof=ddof)
        if axis and mns.ndim < a.ndim:
            res = ((a - np.expand_dims(mns, axis=axis)) /
                   np.expand_dims(sstd, axis=axis))
        else:
            res = (a - mns) / sstd
        return np.nan_to_num(res)
    
    # Time series shifting
    def roll_zeropad(self, a, shift, axis=None):
        a = np.asanyarray(a)
        if shift == 0:
            return a
        if axis is None:
            n = a.size
            reshape = True
        else:
            n = a.shape[axis]
            reshape = False
        if np.abs(shift) > n:
            res = np.zeros_like(a)
        elif shift < 0:
            shift += n
            zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
            res = np.concatenate((a.take(np.arange(n-shift, n), axis), zeros), axis)
        else:
            zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
            res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
        if reshape:
            return res.reshape(a.shape)
        else:
            return res

    #####################################
    # Efficient correlation computations
    #####################################
    def ncc_c(self, x, y):
        den = np.array(norm(x) * norm(y))
        den[den == 0] = np.Inf

        x_len = len(x)
        fft_size = 1 << (2*x_len-1).bit_length()
        cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
        cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
        return np.real(cc) / den

    def ncc_c_3dim(self, x, y):
        """
        Variant of NCCc that operates with 2 dimensional X arrays and 2 dimensional
        y vector
        Returns a 3 dimensional array of normalized fourier transforms
        """
        den = norm(x, axis=1)[:, None] * norm(y, axis=1)
        den[den == 0] = np.Inf
        x_len = x.shape[-1]
        fft_size = 1 << (2*x_len-1).bit_length()
        cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size))[:, None])
        cc = np.concatenate((cc[:,:,-(x_len-1):], cc[:,:,:x_len]), axis=2)
        return np.real(cc) / den.T[:, :, None]
    
    # Get distance between x and y
    # shift y to align with x
    def sbd(self, x, y):
        ncc = self.ncc_c(x, y)
        idx = ncc.argmax()
        dist = 1 - ncc[idx]
        yshift = self.roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

        return dist, yshift
    
    # Perform decomposition of shape matrix S in order to get
    # new centroid
    def get_shape(self, S):
        m = S.shape[0]; 
        
        # If there are no points in the cluster, return flat
        if np.sum(np.abs(S))==0:
            return np.zeros([1, m]); 
        
        O = np.empty((m, m))
        O.fill(1.0/m)
        Q = np.eye(m) - O

        M = np.dot(np.dot(Q, S), Q)
        _, vec = eigh(M)
        centroid = vec[:, -1]
        
        return centroid
        
    
    # Extract the shape using an initial shape
    def extract_shape(self, idx, X, j, cur_center, S_init, weights):
        #------ Get set of points in this cluster, aligned to current center ------#
        _a = []
        _w = []
        # Iterate through all data points
        for i in range(len(idx)):
            # If the data point belongs to the current cluster
            # align it with the current centroid and save it. 
            if idx[i] == j:
                if cur_center.sum() == 0:
                    opt_x = X[i, :]
                else:
                    _, opt_x = self.sbd(cur_center, X[i, :])
                _a.append(opt_x)
                _w.append(weights[i]); 
        A = np.array(_a)
        W = np.array(_w); 
        #--------------------------------------------------------------------------#
        
        # If there are no new points, use the initial center
        if len(A) == 0:
            return self.centers[j, :]; 
        
        # The new shape matrix is the sum of component from new points 
        # and prior shape matrix
        m = A.shape[1]
        X = self.zscore(A, axis=1, ddof=1)
        X = X * W[:, np.newaxis]; 
        S = np.dot(X.transpose(), X) + S_init
        centroid = self.get_shape(S); 

        finddistance1 = math.sqrt(((A[0, :] - centroid) ** 2).sum())
        finddistance2 = math.sqrt(((A[0, :] + centroid) ** 2).sum())

        if finddistance1 >= finddistance2:
            centroid *= -1

        return self.zscore(centroid, ddof=1)
    
    # Initialize cluster membership
    def init_idx(self, X):
        n = X.shape[0]
        if self.init_random:
            idx = randint(0, self.k, size=n)
        else:
            distances = (1 - self.ncc_c_3dim(X, self.centers).max(axis=2)).T
            idx = distances.argmin(1)
        return idx

    ##########################################################################
    # CLUSTERING METHODS
    ##########################################################################
    
    # Inputs
    # X - nxm. n = number of data points, m = Number of time points
    def _kshape(self, X, weights=None, verbose=True):
        # Check time points of new data
        m = X.shape[1]
        if m != self.m:
            raise Exception('Time points mismatch:', m, self.m);
        
        n = X.shape[0]; 
        if weights is None:
            weights = np.ones(n); 
        idx = self.init_idx(X);
        centroids = self.centers.copy(); 
        distances = np.empty((n, self.k))

        for it in range(100):
            old_idx = idx
            for j in range(self.k):
                centroids[j] = self.extract_shape(idx, X, j, centroids[j, :], self.S[j, :], weights)
            # distances has dimensions n x k
            distances = (1 - self.ncc_c_3dim(X, centroids).max(axis=2)).T
            
            idx = distances.argmin(1)
            if verbose:
            # How many indices changed?
                print('Change in clusters', np.sum(old_idx != idx)); 
                
            if np.array_equal(old_idx, idx):
                break

        return idx, centroids

    # Clusters the data in X with the initial clusters in this object
    def kshape(self, X, weights=None):
        idx, centroids = self._kshape(np.array(X), weights=weights)
        clusters = []
        for i, centroid in enumerate(centroids):
            series = []
            for j, val in enumerate(idx):
                if i == val:
                    series.append(j)
            clusters.append((centroid, series))
        return clusters
    
    #######################
    # Probablistic kShape
    #######################
    
    def kshape_prob(self, X, weights=None):
        idx, centroids = self._kshape_prob(np.array(X), weights=weights)
        clusters = []; 
        for i, centroid in enumerate(centroids):
            series = []; 
            for j, val in enumerate(idx):
                if i == val:
                    series.append(j)
            clusters.append((centroid, series))
        return clusters
    
    # Inputs
    # X - nxm. n = number of data points, m = Number of time points
    def _kshape_prob(self, X, weights=None, verbose=True):
        # Check time points of new data
        m = X.shape[1]
        if m != self.m:
            raise Exception('Time points mismatch:', m, self.m);
        
        n = X.shape[0]; 
        if weights is None:
            weights = np.ones(n); 
        # Initialize cluster memberships
        idx = self.init_idx(X);
        # Copy current centroids
        centroids = self.centers.copy(); 
        # Initialize cluster distance means
        means = np.zeros(self.k); 
        # Initialize cluster distance variances
        sigs = np.ones(self.k); 
        # Initialize distances to clusters
        distances = np.empty((n, self.k))
        
        n_it = 100; 
        mean_evolution = np.zeros((self.k, n_it)); 
        sigs_evolution = np.zeros((self.k, n_it)); 
        for it in range(n_it):
            old_idx = idx
            for j in range(self.k):
                centroids[j] = self.extract_shape(idx, X, j, centroids[j, :], self.S[j, :], weights)
            # distances has dimensions n x k
            distances = (1 - self.ncc_c_3dim(X, centroids).max(axis=2)).T
            
            # Get updated means and std deviations of cluster distances
            means, sigs = self.update_cluster_stats(idx, means, sigs, distances); 
            # Record means and std deviations of clusters
            mean_evolution[:, it] = means; 
            sigs_evolution[:, it] = sigs; 
            
            # Compute probability of each cluster membership & assign to most likely cluster
            idx = self.prob_idx(means, sigs, distances); 
            #idx = distances.argmin(1)
            
            
            if verbose:
            # How many indices changed?
                print('Cluster means', means); 
                print('Cluster Std Dev', sigs); 
                #print('Change in clusters', np.sum(old_idx != idx)); 
                #print('Number of outliers', np.sum(idx == self.k)); 
                #print('Indices', idx); 
                
            if np.array_equal(old_idx, idx):
                mean_evolution = mean_evolution[:, 0:it+1]; 
                sigs_evolution = sigs_evolution[:, 0:it+1]; 
                break

        return idx, centroids, mean_evolution, sigs_evolution
    
    #####################################################################
    
    # Updates the clusters of this object with new data. 
    def _update_clusters(self, clusters, X, weights=None):
        n = X.shape[0];
        if weights is None:
            weights = np.ones(n); 
        else:
            weights = np.array(weights); 
        for i in range(self.k):
            centroid = clusters[i][0]; 
            idxs = clusters[i][1];

            self.centers[i, :] = centroid
            Xi = X[idxs, :] * weights[idxs, np.newaxis]; 
            self.S[i, :, :] = self.S[i, :, :] + np.dot(Xi.transpose(), Xi)
            self.counts[i] += np.sum(weights[idxs]); 
            
    def add(self, in_data):
        # If input is a kshape object, we need to get the centers and weights
        if isinstance(in_data, kShapeStream):
            k2 = in_data.k; 
            m2 = in_data.m; 
            X = []
            weights = [] 
            for i in range(k2):
                if np.sum(np.abs(in_data.centers[i, :])) > 0:
                    X.append(in_data.centers[i, :]); 
                    weights.append(in_data.counts[i]); 
            X = np.array(X); 
            weights = np.array(weights); 
        # If input is an array, we pass it directly
        elif isinstance(in_data, np.ndarray):
            n = in_data.shape[0];
            m2 = in_data.shape[1]; 
            weights = np.ones(n); 
            X = in_data; 
        
        # Check for size match
        if m2 != self.m:
            raise Exception('Data sizes mismatch', self.m, m2);
            
        clusters = self.kshape(X, weights=weights); 
        self._update_clusters(clusters, X, weights=weights); 
        
        # Don't randomly initialize next batch
        self.init_random = False;
        return clusters
    
    ###############################################################################
    # PROBABILITY METHODS
    ###############################################################################
    def getDist(self, i):
        mu = self.centers[i, :]; 
        c = self.counts[i];
        if c==0:
            var = np.zeros(self.m); 
        else:
            var = (1.0/self.counts[i])*np.diagonal(self.S[i, :, :]) - mu**2; 
        return mu, np.abs(var)
    
    def dist_outliers(self, distances, idxs, thresh=2):
    # distances - n x k array of distances between each data point and centroid
    # idxs - n array of cluster label for each data point
        n = np.size(idxs); 
        new_idxs = idxs.copy(); 
        # Iterate through clusters
        for i in range(self.k):
            # Get distance statistics for this cluster
            disti = distances[idxs==i, :]; 
            mu = np.mean(disti); sig = np.sqrt(np.var(disti)); 
            # Iterate through data points
            for j in range(n): 
                if idxs[j]==i:
                    if np.abs(distances[j, i]-mu) > thresh*sig:
                        # This is an outlier
                        new_idxs[j] = self.k;
        return new_idxs
    
    def prob_idx(self, means, sigs, distances, thresh=2): 
    # Inputs
    # means - k array of mean distances of each cluster
    # sigs - k array of std. dev. of distances of each cluster
    # distances - n x k array of distances between each data point and cluster
    # Outputs
    # idxs - n array containing values 0 to k indicating cluster
    # membership (0-(k-1)) or outlier (k). 
        n = np.shape(distances)[0]; 
        # Get the standard deviation for all distances
        sig_dist = np.abs(distances-means)/sigs; 
        print('Sigma distances', sig_dist); 
        # Initialize cluster membership array
        idxs = np.zeros(n); 
        for i in range(n):
            # Find the maximum likelihood cluster
            min_dist = np.argmin(sig_dist[i, :]); 
            if sig_dist[i, min_dist] > thresh: 
                idxs[i] = self.k; 
                print('Outlier', i, 'with distances: ', sig_dist[i, :], 'mindist', min_dist, 'thresh', thresh); 
            else:
                idxs[i] = min_dist; 
        return idxs; 
    
    def update_cluster_stats(self, idxs, means, sigs, distances):
        means = np.zeros(self.k); sigs = np.ones(self.k); 
        for i in range(self.k):
            n_i = np.sum(idxs==i); 
            if n_i>0:
                dists_i = distances[idxs==i, i]; 
                weight = n_i/(1+n_i); 
                means[i] = np.mean(dists_i); sigs[i] = weight*np.sqrt(np.var(dists_i)) + (1-weight)*1; 
        return means, sigs 
        
    ###############################################################################
    # VISUALIZATION METHODS
    ###############################################################################
    
    # Visualize clusters and data
    def visualize(clusters, X, cols=5):
        k = len(clusters); 
        n  = X.shape[0]; 
        
        rows = int(np.ceil(k / cols)); 
        plt.figure(figsize=(cols*3, rows*3 + 2))
        
        for i in range(k):
            plt.subplot(rows, cols, i+1)
            centroid, els = clusters[i];
            if len(els) != 0:
                plt.plot(X[els, :].T, "k-", alpha=.2, linewidth=3)
            plt.plot(centroid, "r-", linewidth=3)
            plt.title("Cluster %d : %d" % (i, len(els)), fontsize=20)
            
    # Visualize current centers and variance
    def visualizeCenters(self, cols=5):
        rows = int(np.ceil(self.k / cols)); 
        plt.figure(figsize=(cols*3, rows*3 + 2))
        x = np.arange(0, self.m); 
        for i in range(self.k):
            plt.subplot(rows, cols, i+1); 
            
            # Get center and variance
            mu, var = self.getDist(i); 
            # Plot
            plt.fill_between(x, mu-np.sqrt(var), mu+np.sqrt(var), color='gray', alpha=0.5); 
            plt.plot(x, mu, 'b-', linewidth=5);
            plt.title("Cluster %d : %d" % (i, self.counts[i]), fontsize=20) 