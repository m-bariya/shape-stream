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

#########################################################
# Functions called by clustering methods

# Normalize matrix of time series
def zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    # Compute mean and std dev along axis
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd

    return np.nan_to_num(res)

# Time series shifting
# Translate each time series in a by shift
def roll_zeropad(a, shift, axis=None):
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
    
#---------------------------------------------------------
# Efficient correlation computation
def ncc_c(x, y):
        den = np.array(norm(x) * norm(y))
        den[den == 0] = np.Inf

        x_len = len(x)
        fft_size = 1 << (2*x_len-1).bit_length()
        cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
        cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
        return np.real(cc) / den

def ncc_c_3dim(x, y):
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
def sbd(x, y):
    # Find correlation between x & y for different shifts
    # ncc is a vector of distances
    ncc = ncc_c(x, y)
    
    # Find time shift corresponding to maximum correlation
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    
    # Shift y to align with x
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return dist, yshift


#########################################################
class kShapeStream:
# The base kShape Clustering Stream. 
# Distances are 1-correlation
    
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
    
    # Initialize cluster membership for timeseries in X
    # X - [n x m]
    def init_idx(self, X):
        n = X.shape[0]
        # Assign randomly
        if self.init_random:
            idx = randint(0, self.k, size=n)
        # Assign to nearest cluster
        else:
            distances = (1 - ncc_c_3dim(X, self.centers).max(axis=2)).T
            idx = distances.argmin(1)
        return idx
    
    #-----------------------------------------------------------------
    # Shape extraction methods
    
    # Perform decomposition of shape matrix S 
    # to get new cluster centroid
    # S - [m x m]
    def get_shape(self, S):
        m = self.m
        
        # Centroid of empty cluster is flat
        if (np.linalg.norm(S)==0):
            return np.zeros([1, m]); 
        
        # Eq 14 in kShape paper
        O = np.empty((m, m))
        O.fill(1.0/m)
        Q = np.eye(m) - O

        M = np.dot(np.dot(Q, S), Q)
        _, vec = eigh(M)
        centroid = vec[:, -1]
        
        return centroid
        
    
    # Extract new center from previous shape matrix and new member time series
    # idx - [n] array of cluster indices for time series in X
    # X - [n x m] array of new time series
    # j - [int] cluster of interest
    # cur_center - [m] current centroid of cluster j
    # S_init - [m x m] Initial shape matrix of cluster j
    # weights - [n] weights associated with each new cluster member
    def extract_shape(self, idx, X, j, cur_center, S_init, weights):
        # Get time series in X belonging to cluster j
        _a = []
        _w = []
        # Iterate through all data points
        for i in range(len(idx)):
            # If the data point belongs to the current cluster
            # align it with the current centroid and save it. 
            if idx[i] == j:
                # If current center is empty, no alignment
                if cur_center.sum() == 0:
                    opt_x = X[i, :]
                else:
                    _, opt_x = sbd(cur_center, X[i, :])
                # Save aligned timeseries and weights
                _a.append(opt_x)
                _w.append(weights[i]); 
        # [- x m] matrix of new time points in cluster j
        A = np.array(_a)
        # [-] array of weights for each new point in j
        W = np.array(_w); 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # If there are no new points, use the initial center
        if len(A) == 0:
            return self.centers[j, :]; 
        
        # The new shape matrix is the sum of component from new points 
        # and prior shape matrix
        # Normalize timeseries
        Xp = zscore(A, axis=1, ddof=1)
        Xp = Xp * W[:, np.newaxis]; 
        # Combined shape matrix
        S = np.dot(Xp.transpose(), Xp) + S_init
        # Compute centroid
        centroid = self.get_shape(S); 
        
        # Shape extraction is specified up to a sign. We determine
        # the correct sign after shape extraction. 
        finddistance1 = math.sqrt(((A[0, :] - centroid) ** 2).sum())
        finddistance2 = math.sqrt(((A[0, :] + centroid) ** 2).sum())
        if finddistance1 >= finddistance2:
            centroid *= -1
        # Return normalized centroid
        return zscore(centroid, ddof=1)

    #-----------------------------------------------------------------
    # CLUSTERING METHOD
    
    # Clustering wrapper function
    # X - matrix of timeseries data to cluster
    # Returns list of (centroid, index_list) pairs
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
    
    # Clustering loop
    # X - [n x m] matrix of timeseries data
    # n - number of timeseries, m - number of time points
    # weights - [n, optional] weights for each timeseries
    def _kshape(self, X, weights=None, verbose=True):
        # Check time points of new data
        m = X.shape[1]
        if m != self.m:
            raise Exception('Time points mismatch:', m, self.m);
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Initialize
        n = X.shape[0]; 
        if weights is None:
            weights = np.ones(n); 
        idx = self.init_idx(X);
        centroids = self.centers.copy(); 
        distances = np.empty((n, self.k))

        for it in range(100):
            old_idx = idx
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 2. Shape Extraction
            for j in range(self.k):
                centroids[j] = self.extract_shape(idx, X, j, centroids[j, :], self.S[j, :], weights)
            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 3. Cluster Assignment
            # distances has dimensions n x k
            distances = (1 - ncc_c_3dim(X, centroids).max(axis=2)).T
            idx = distances.argmin(1)
                
            # If no change in assignment, return
            if np.array_equal(old_idx, idx):
                break

        return idx, centroids
    
    # ----------------------------------------------------
    # STREAMING METHODS 
    # Updates the clusters of this object with new data. 
    
    # Update existing clusters with new timeseries
    # clusters - list of (centroid, indices) for each cluster
    # X - [n x m] matrix of data
    # weights - [n] optional weights for each data point
    def _update_clusters(self, clusters, X, weights=None):
        n = X.shape[0];
        
        # If no weights, are given, we assume no weighting
        if weights is None:
            weights = np.ones(n); 
        else:
            weights = np.array(weights); 
            
        # Iteratively update clusters
        for i in range(self.k):
            centroid = clusters[i][0]; 
            idxs = clusters[i][1];
            
            # Update centroid
            self.centers[i, :] = centroid
            # Update shape matrix
            Xi = X[idxs, :] * weights[idxs, np.newaxis]; 
            self.S[i, :, :] = self.S[i, :, :] + np.dot(Xi.transpose(), Xi)
            # Update cluster counts
            self.counts[i] += np.sum(weights[idxs]); 
    
    # Add new data to existing clusters, or merge two cluster sets
    # in_data - [n x m array or kShapeStream object]
    def add(self, in_data):
        # If in_data is kShapeStream, add its centers and weights to 
        # this kShapeStream
        if isinstance(in_data, kShapeStream):
            # Check for size match
            k2 = in_data.k; m2 = in_data.m; 
            if m2 != self.m:
                raise Exception('Data sizes mismatch', self.m, m2);
            # Get centroids and weights
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
            n = in_data.shape[0]; m2 = in_data.shape[1]; 
            if m2 != self.m:
                raise Exception('Data sizes mismatch', self.m, m2);
            weights = np.ones(n); 
            X = in_data; 
        
        # Add new centroids/data to existing clusters
        clusters = self.kshape(X, weights=weights); 
        self._update_clusters(clusters, X, weights=weights); 
        
        # Don't randomly initialize next batch
        self.init_random = False;
        return clusters
        
    # ----------------------------------------------------
    # VISUALIZATION METHODS
    
    def visualize(self, clusters, X, cols=None):
        if cols is None:
            cols = self.k
        # Create figure
        rows = int(np.ceil(self.k / cols)); 
        plt.figure(figsize=(cols*3, rows*3 + 2))
        for i in range(self.k):
            plt.subplot(rows, cols, i+1); 
            centroid, els = clusters[i];
            if len(els) != 0:
                for j in els:
                    _, opt_x = sbd(centroid, X[j, :])
                    plt.plot(opt_x, "k-", alpha=.2, linewidth=3)
            plt.plot(centroid, "b-", linewidth=3)
            plt.title("Members : %d" % (self.counts[i]), fontsize=20)
            
            
#########################################################
class kShapeProbStream(kShapeStream):
# Probability based kShape streaming clustering
# Points are added to clusters based on likelihood of distance
    def __init__(self, k, m):
        super().__init__(k, m);
        # Distance statistics of clusters
        self.means = np.zeros(k); 
        self.sigs = np.ones(k); 
        self.dists2 = np.ones(k);
    
    # Clustering wrapper function
    # X - matrix of timeseries data to cluster
    # Returns list of (centroid, index_list) pairs
    def kshape(self, X, weights=None):
        idx, centroids = self._kshape(np.array(X), weights=weights)
        clusters = []; 
        for i, centroid in enumerate(centroids):
            series = []; 
            for j, val in enumerate(idx):
                if i == val:
                    series.append(j)
            clusters.append((centroid, series))
        # Append outliers at the end
        outliers = idx[idx==self.k]
        clusters.append(outliers); 
        return clusters
    
    # Clustering loop
    # X - [n x m] matrix of timeseries data
    # n - number of timeseries, m - number of time points
    # weights - [n, optional] weights for each timeseries
    def _kshape(self, X, weights=None, verbose=True):
        
        # Check time points of new data
        m = X.shape[1]
        if m != self.m:
            raise Exception('Time points mismatch:', m, self.m);
        
        n = X.shape[0]; 
        # If no weights given, assume unweighted
        if weights is None:
            weights = np.ones(n); 
            
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Initialize
        # Initialize cluster memberships
        idx = self.init_idx(X);
        # Initialize distances to clusters
        distances = np.empty((n, self.k))
        # Initialize cluster distance means
        means = self.means.copy(); 
        # Initialize cluster distance variances
        sigs = self.sigs.copy(); 
        # Save current centroids
        centroids = self.centers.copy(); 
        
        for it in range(100):
            old_idx = idx
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 2. Shape Extraction
            for j in range(self.k):
                centroids[j] = self.extract_shape(idx, X, j, centroids[j, :], self.S[j, :], weights)
                
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 3. Cluster Assignment
            # distances has dimensions n x k
            distances = (1 - ncc_c_3dim(X, centroids).max(axis=2)).T
            # Update means and std deviations of cluster distances
            means, sigs, dists2 = self.update_cluster_stats(idx, distances); 
            # Assign to most likely clusters
            idx = self.prob_idx(means, sigs, distances); 
            # If no change in assignment, return
            if np.array_equal(old_idx, idx): 
                break

        return idx, centroids
    
    
    # Add new data to existing clusters, or merge two cluster sets
    # in_data - [n x m array or kShapeStream object]
    def add(self, in_data):
        # If in_data is kShapeStream, add its centers and weights to 
        # this kShapeStream
        if isinstance(in_data, kShapeStream):
            # Check for size match
            k2 = in_data.k; m2 = in_data.m; 
            if m2 != self.m:
                raise Exception('Data sizes mismatch', self.m, m2);
            # Get centroids and weights
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
            n = in_data.shape[0]; m2 = in_data.shape[1]; 
            if m2 != self.m:
                raise Exception('Data sizes mismatch', self.m, m2);
            weights = np.ones(n); 
            X = in_data; 
        
        # Add new centroids/data to existing clusters
        clusters = self.kshape(X, weights=weights); 
        self._update_clusters(clusters, X, weights=weights); 
        
        # Don't randomly initialize next batch
        self.init_random = False;
        return clusters
    
    def _update_clusters(self, clusters, X, weights=None):
        n = X.shape[0];
        
        # If no weights, are given, we assume no weighting
        if weights is None:
            weights = np.ones(n); 
        else:
            weights = np.array(weights); 
            
        # Iteratively update clusters
        for i in range(self.k):
            centroid = clusters[i][0]; 
            idxs = clusters[i][1];
            
            # Update centroid
            self.centers[i, :] = centroid
            # Get distances of X to centroids
            distances = (1 - ncc_c_3dim(X, np.expand_dims(centroid, axis=1)).max(axis=2)).T
            # Update cluster statistics
            means, sigs, dists2 = self.update_cluster_stats(idxs, distances)
            self.means = means; self.sigs = sigs; self.dists2 = dists2; 
            # Update shape matrix
            Xi = X[idxs, :] * weights[idxs, np.newaxis]; 
            self.S[i, :, :] = self.S[i, :, :] + np.dot(Xi.transpose(), Xi)
            # Update cluster counts
            self.counts[i] += np.sum(weights[idxs]); 
        
    # ----------------------------------------------------
    # PROBABILITY DISTANCES
    
    def prob_idx(self, means, sigs, distances, thresh=1.5): 
    # means - [k] array of mean dist to center in each cluster
    # sigs - [k] array of std. dev. of dist to center
    # distances - n x k array of dists between each data point and cluster
    # Returns
    # idxs - [n] array of assigned cluster index
    # membership (0-(k-1)) or outlier (k). 
        n = np.shape(distances)[0]; 
        
        # Convert distances to number of std devs from mean
        sig_dist = np.abs(distances-means)/sigs; 
        print(sig_dist)
        # Initialize cluster membership array
        idxs = np.zeros(n); 
        for i in range(n):
            # Find the maximum likelihood cluster
            min_dist = np.argmin(sig_dist[i, :]); 
            if sig_dist[i, min_dist] < thresh: 
                idxs[i] = min_dist
            else:
                # Outlier
                print('OUTLIER'); 
                idxs[i] = self.k; 
        return idxs; 
    
    # Update means and std devs of dists to centroid for each cluster
    def update_cluster_stats(self, idxs, distances):
        # Start with previous means / variances / dist2 of clusters
        means = self.means.copy(); sigs = self.sigs.copy(); dists2 = self.dists2.copy(); 
        # Iterate through clusters and update stats
        for i in range(self.k):
            # Find total number of points in the cluster
            nprev = self.counts[i]; ncur = np.sum(idxs==i); 
            ntot = nprev + ncur; 
            # If points have been added (in this dataset), update stats
            if ncur>0:
                d = distances[idxs==i, i]; 
                # Mean is the weighted average of previous and current mean
                means[i] = (means[i]*nprev + np.sum(d)) / ntot; 
                # Compute squared distance
                dists2[i] = (dists2[i]*nprev + np.sum(d**2)) / ntot; 
                # Sigmas are computed with a weighting, to increase 
                # certainty with ntot
                sig = np.sqrt(dists2[i]-means[i]**2)
                weight = ntot / (1 + ntot); 
                sigs[i] = weight*sig + (1-weight)*1; 
        return means, sigs , dists2
    
    # ----------------------------------------------------
    # VISUALIZATION METHODS
    
    # This differs from that of the superclass, in that we show outliers
    def visualize(self, clusters, X, cols=None):
        if cols is None:
            cols = self.k + 1
        # Create figure
        rows = int(np.ceil(self.k + 1 / cols)); 
        plt.figure(figsize=(cols*3, rows*3 + 2))
        for i in range(self.k):
            plt.subplot(rows, cols, i+1); 
            centroid, els = clusters[i];
            if len(els) != 0:
                for j in els:
                    _, opt_x = sbd(centroid, X[j, :])
                    plt.plot(opt_x, "k-", alpha=.2, linewidth=3)
            plt.plot(centroid, "b-", linewidth=3)
            plt.title("Members : %d" % (self.counts[i]), fontsize=20)
        # Plot outlier
        plt.subplot(rows, cols, self.k+1);
        els = clusters[self.k]
        if len(els) != 0:
            for j in els:
                plt.plot(X[j, :], linewidth=3); 
        plt.title("Outliers : %d" % len(els), fontsize=20); 