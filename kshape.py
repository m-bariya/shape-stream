##################################################################################################################
##################################################################################################################
'''
This file contains two streaming time series clustering algorithms based on kShape.
1. kShapeStream - A simple streaming version of kShape
2. kShapeProbStream - A modification of [1.] to assign points to a cluster based on a probablistic distance
'''
##################################################################################################################
##################################################################################################################
# Imports
import math
import numpy as np
from numpy.random import randint
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


''' Helper Functions used in clustering. '''

def zscore(a, axis=0, ddof=0):
    ''' Normalize matrix of time series in a '''
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

def roll_zeropad(a, shift, axis=None):
    ''' Shift time series in a by 'shift', filling with zeros. '''
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

def ncc_c(x, y):
    """
    Efficiently compute cross correlation between x and y vectors using ffts.
    Returns array of cross correlation.
    x - 1 x m, y - 1 x m.
    returns - m x 1
    """
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
    y vector. Returns a 3 dimensional array of normalized fourier transforms
    x - n x m, y - k x m
    returns - n x k x m
    """
    den = norm(x, axis=1)[:, None] * norm(y, axis=1)
    den[den == 0] = np.Inf
    x_len = x.shape[-1]
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size))[:, None])
    cc = np.concatenate((cc[:,:,-(x_len-1):], cc[:,:,:x_len]), axis=2)
    return np.real(cc) / den.T[:, :, None]

def sbd(x, y):
    """
    Shape Based Distance between vectors x & y
    Shifts y to align with x by maximizing correlation
    x - 1 x m, y - 1 x m
    """
    # ncc is a vector of distances for different shifts
    ncc = ncc_c(x, y)

    # Find time shift corresponding to maximum correlation
    idx = ncc.argmax()
    dist = 1 - ncc[idx]

    # Shift y to align with x
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return dist, yshift

def gaussian(x, mu, sig):
    """
    Returns a gaussian curve with given mean and std dev. Used in visualization.
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


##################################################################################################################
##################################################################################################################
"""
kShapeStream: A Streaming version of the kShape Algorithm. 
Distances between data points are the sbd. Points assigned to nearest cluster. 
"""
##################################################################################################################
##################################################################################################################
class kShapeStream:

    def __init__(self, k, m):
        """
        Initialize stream with k clusters of length m time series.
        k - scalar; number of clusters, m - scalar; time series length
        """
        # Number of clusters
        self.k = k
        # Number of times points
        self.m = m
        # Centroids : Initially zero
        self.centers = np.zeros([k, m])
        # Counts of elements in each cluster
        self.counts = np.zeros([k])
        # Shape Matrices from which centroids
        # are extracted.
        self.S = np.zeros([k, m, m])
        # How to initialize cluster membership
        self.init_random = True

        # Recorder - Shows evolution of clusters over iterations
        self.record = False
        self.recorder = {"centroids" : np.zeros([k, m, 1]), "counts" : np.zeros([k, 1])}

        # Visualization specifications
        self.ps = {};
        self.ps['background_color'] = 'whitesmoke'
        self.ps['centroid_size'] = 3
        self.ps['member_color'] = 'gray'
        self.ps['member_alpha'] = 0.2
        self.ps['member_size'] = 2
        self.ps['centroid_colors'] = pl.cm.jet(np.linspace(0, 1, self.k))
        self.ps['show_x'] = False; self.ps['show_y'] = False
        self.ps['count_size'] = 16;


    def init_idx(self, X):
        """
        Initialize cluster membership of n time series in X either randomly or to nearest cluster center.
        X - n x m
        """
        n = X.shape[0]
        # Assign randomly
        if self.init_random:
            idx = randint(0, self.k, size=n)
        # Assign to nearest cluster
        else:
            distances = (1 - ncc_c_3dim(X, self.centers).max(axis=2)).T
            idx = distances.argmin(1)
        return idx

    """
    Recorder methods. Recorder optionally tracks cluster evolution. 
    """
    def reset_record(self):
        self.recorder = {"centroids" : np.zeros([self.k, self.m, 1]), "counts" : np.zeros([self.k, 1])}

    def update_record(self, dict_vals):
        for key, val in dict_vals.items():
            self.recorder[key] = np.concatenate([self.recorder[key], val[..., np.newaxis]], axis=-1)

    def get_counts(self, idx):
        """
        Counts members in each cluster for assignments in idx.
        idx - n x 1.
        returns - k x 1.
        """
        counts = np.zeros(self.k)
        for i in range(self.k):
            counts[i] = np.sum(idx==i)
        return counts

    """
    Shape extraction methods 
    update shape matrices & obtain centroids from cluster members.
    """

    def get_shape(self, S):
        """
        Perform decomposition of shape matrix S to get new cluster centroid
        S - m x m
        returns -  m x 1
        """
        m = self.m

        # Centroid of empty cluster is zeros
        if (np.linalg.norm(S)==0):
            return np.zeros([1, m])

        # Eq 14 in kShape paper
        O = np.empty((m, m))
        O.fill(1.0/m)
        Q = np.eye(m) - O

        M = np.dot(np.dot(Q, S), Q)
        _, vec = eigh(M)
        centroid = vec[:, -1]

        return centroid

    def extract_shape(self, idx, X, j, cur_center, S_init, weights):
        """
        Find new centroid of cluster j from its previous shape matrix and new members
        idx - n. cluster assignments of time series in X
        X - n x m. new time series.
        j - int. cluster of interest.
        cur_center - m. current centroid of cluster j
        S_init - m x m. initial shape matrix of cluster j
        weights - n. weights associated with each time series of X. (used only if combining two kshapestreams)
        """
        # Get time series in X belonging to cluster j & corresponding weights
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
                _w.append(weights[i])
        # [- x m] matrix of new time points in cluster j
        A = np.array(_a)
        # [-] array of weights for each new point in j
        W = np.array(_w)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # If there are no new points, use the initial center
        if len(A) == 0:
            return self.centers[j, :]

        # The new shape matrix is the sum of component from new points
        # and prior shape matrix
        # Normalize timeseries
        Xp = zscore(A, axis=1, ddof=1)
        Xp = Xp * W[:, np.newaxis]
        # Combined shape matrix
        S = np.dot(Xp.transpose(), Xp) + S_init
        # Compute centroid
        centroid = self.get_shape(S)

        # Shape extraction is specified up to a sign. We determine
        # the correct sign after shape extraction.
        finddistance1 = math.sqrt(((A[0, :] - centroid) ** 2).sum())
        finddistance2 = math.sqrt(((A[0, :] + centroid) ** 2).sum())
        if finddistance1 >= finddistance2:
            centroid *= -1
        # Return normalized centroid
        return zscore(centroid, ddof=1)

    """ 
    Clustering Methods 
    """

    def kshape(self, X, weights=None):
        """
        Clustering wrapper function
        X - n x m. Time series data to cluster
        Returns clusters as list of (centroid, index_list) pairs
        """
        idx, centroids = self._kshape(np.array(X), weights=weights)
        clusters = []
        for i, centroid in enumerate(centroids):
            series = []
            for j, val in enumerate(idx):
                if i == val:
                    series.append(j)
            clusters.append((centroid, series))
        return clusters

    def _kshape(self, X, weights=None, verbose=True):
        """
        Clustering loop
        X - n x m matrix of n m-length time series data
        weights - n (optional). weights for each time series (used only if combining two kshapestreams)
        """
        # Check time points of new data
        m = X.shape[1]
        if m != self.m:
            raise Exception('Time points mismatch:', m, self.m)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Initialize
        n = X.shape[0]
        if weights is None:
            weights = np.ones(n)
        idx = self.init_idx(X)
        centroids = self.centers.copy()
        distances = np.empty((n, self.k))

        # Iterate
        for it in range(100):
            old_idx = idx
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 2. Shape Extraction
            for j in range(self.k):
                centroids[j] = self.extract_shape(idx, X, j, centroids[j, :], self.S[j, :], weights)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 3. Cluster Assignment
            # distances has dimensions n x k
            distances = (1 - ncc_c_3dim(X, centroids).max(axis=2)).T
            idx = distances.argmin(1)

            # Record
            if self.record:
                counts = self.get_counts(idx)
                self.update_record({"centroids" : centroids, "counts" : counts})

            # If no change in assignment, return
            if np.array_equal(old_idx, idx):
                break

        return idx, centroids

    """
    Streaming Methods.
    """

    def _update_clusters(self, clusters, X, weights=None):
        """
        Update the clusters of this kshapestream with new data.
        clusters - list of (centroid, indices) for each cluster
        X - n x m matrix of data
        weights - n. optional weights for each time series in X
        """
        n = X.shape[0]

        # If no weights, are given, we assume no weighting
        if weights is None:
            weights = np.ones(n)
        else:
            weights = np.array(weights)

        # Iteratively update clusters
        for i in range(self.k):
            centroid = clusters[i][0]
            idxs = clusters[i][1]

            # Update centroid
            self.centers[i, :] = centroid
            # Update shape matrix
            Xi = X[idxs, :] * weights[idxs, np.newaxis]
            self.S[i, :, :] = self.S[i, :, :] + np.dot(Xi.transpose(), Xi)
            # Update cluster counts
            self.counts[i] += np.sum(weights[idxs])

    def add(self, in_data):
        """
        Add new data or merge other clusters with clusters of this kshapestream
        in_data - n x m array OR kShapeStream object
        """

        # If in_data is another kShapeStream, add its centers and weights to
        # this kShapeStream
        if isinstance(in_data, kShapeStream):
            # Check for size match
            k2 = in_data.k; m2 = in_data.m
            if m2 != self.m:
                raise Exception('Data sizes mismatch', self.m, m2)
            # Get centroids and weights
            X = []
            weights = []
            for i in range(k2):
                if np.sum(np.abs(in_data.centers[i, :])) > 0:
                    X.append(in_data.centers[i, :])
                    weights.append(in_data.counts[i])
            X = np.array(X)
            weights = np.array(weights)

        # If input is an array, we pass it directly
        elif isinstance(in_data, np.ndarray):
            n = in_data.shape[0]; m2 = in_data.shape[1]
            if m2 != self.m:
                raise Exception('Data sizes mismatch', self.m, m2)
            weights = np.ones(n)
            X = in_data

        # Add new centroids/data to existing clusters
        clusters = self.kshape(X, weights=weights)
        self._update_clusters(clusters, X, weights=weights)

        # Don't randomly initialize next batch
        self.init_random = False
        return clusters

    """ 
    Visualization Method
    """
    def visualize(self, clusters, X, axes):
        """
        Visualize clusters
        clusters - list of (centroid, indices) for each cluster
        X - n x m of clustered data
        cols - number of columns to plot
        """
        # Check that we have enough axes
        if len(axes) != self.k:
            print('Error: Not enough axes')
            return
        colors = self.ps['centroid_colors']
        for i in range(self.k):
            ax = axes[i]
            centroid, els = clusters[i]
            if len(els) != 0:
                for j in els:
                    _, opt_x = sbd(centroid, X[j, :])
                    ax.plot(opt_x, self.ps['member_color'], alpha=self.ps['member_alpha'], linewidth=self.ps['member_size'])
            ax.plot(centroid, color=colors[i], linewidth=5)
            # Formatting the plot
            # Number of points in cluster
            ax.text(0.9, 0.9, str(self.counts[i]), horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes, fontsize=self.ps['count_size'])
            # Remove ticks
            if not self.ps['show_x']:
                ax.get_xaxis().set_ticks([])
            if not self.ps['show_y']:
                ax.get_yaxis().set_ticks([])
            ax.set_facecolor(self.ps['background_color'])

##################################################################################################################
##################################################################################################################
"""
kShapeProbStream: A probablistic version of kShapeStream. 
Distances between data points are the sbd. Points assigned to most likely cluster, with outliers separated. 
"""

class kShapeProbStream(kShapeStream):
    def __init__(self, k, m):
        """
        Initialize stream with k clusters of length m time series.
        k - scalar; number of clusters, m - scalar; time series length
        """
        super().__init__(k, m)
        # Distance statistics of clusters
        self.means = np.zeros(k)
        self.sigs = np.ones(k)
        self.dists2 = np.ones(k)

        # Threshold for outliers (number of std. deviations)
        self.thresh = 2

        # Recorder
        self.recorder["means"] = np.zeros([k, 1])
        self.recorder["sigs"] = np.zeros([k, 1])

        # Visualization specifications
        self.ps['dist_color'] = 'orangered'
        self.ps['dist_alpha'] = 0.5

    """
    Recorder methods. Recorder optionally tracks cluster evolution. 
    """
    def reset_record(self):
        self.recorder = {"centroids" : np.zeros([self.k, self.m, 1]), "counts" : np.zeros([self.k, 1])}
        self.recorder["means"] = np.zeros([self.k, 1])
        self.recorder["sigs"] = np.zeros([self.k, 1])

    def kshape(self, X, weights=None):
        """
        Clustering wrapper function
        X - n x m. Time series data to cluster
        Returns clusters as list of (centroid, index_list) pairs with last element a list of outlier idxs.
        """
        idx, centroids = self._kshape(np.array(X), weights=weights)
        clusters = []
        for i, centroid in enumerate(centroids):
            series = []
            for j, val in enumerate(idx):
                if i == val:
                    series.append(j)
            clusters.append((centroid, series))
        # Append outliers at the end
        outliers = np.where(idx==self.k)
        clusters.append(outliers)
        return clusters

    def _kshape(self, X, weights=None, verbose=True):
        """
        # Clustering loop
        # X - n x m matrix of n m-length time series data
        # n - number of time series, m - number of time points
        # weights - n (optional). weights for each time series (used only if combining two kshapestreams)
        """
        # Check time points of new data
        m = X.shape[1]
        if m != self.m:
            raise Exception('Time points mismatch:', m, self.m)

        n = X.shape[0]
        # If no weights given, assume unweighted
        if weights is None:
            weights = np.ones(n)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1. Initialize
        # Initialize cluster memberships
        idx = self.init_idx(X)
        # Initialize distances to clusters
        distances = np.empty((n, self.k))
        # Initialize cluster distance means
        means = self.means.copy()
        # Initialize cluster distance variances
        sigs = self.sigs.copy()
        # Save current centroids
        centroids = self.centers.copy()

        for it in range(100):
            old_idx = idx
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 2. Shape Extraction
            for j in range(self.k):
                centroids[j] = self.extract_shape(idx, X, j, centroids[j, :], self.S[j, :], weights)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 3. Cluster Assignment
            # find n x k distances given new centroids
            distances = (1 - ncc_c_3dim(X, centroids).max(axis=2)).T
            # Update means and std deviations of cluster distances
            means, sigs, dists2 = self.update_cluster_stats(idx, distances)

            # Assign to most likely clusters
            idx = self.prob_idx(means, sigs, distances)

            # Record
            if self.record:
                counts = self.get_counts(idx)
                self.update_record({"centroids" : centroids, "counts" : counts, "means" : means, "sigs" : sigs})

            # If no change in assignment, return
            if np.array_equal(old_idx, idx):
                break

        return idx, centroids

    def add(self, in_data):
        """
        Add new data or merge other clusters with clusters of this kshapestream
        in_data - n x m array OR kShapeStream object
        """
        # If in_data is kShapeStream, add its centers and weights to
        # this kShapeStream
        if isinstance(in_data, kShapeStream):
            # Check for size match
            k2 = in_data.k; m2 = in_data.m
            if m2 != self.m:
                raise Exception('Data sizes mismatch', self.m, m2)
            # Get centroids and weights
            X = []
            weights = []
            for i in range(k2):
                if np.sum(np.abs(in_data.centers[i, :])) > 0:
                    X.append(in_data.centers[i, :])
                    weights.append(in_data.counts[i])
            X = np.array(X)
            weights = np.array(weights)

        # If input is an array, we pass it directly
        elif isinstance(in_data, np.ndarray):
            n = in_data.shape[0]; m2 = in_data.shape[1]
            if m2 != self.m:
                raise Exception('Data sizes mismatch', self.m, m2)
            weights = np.ones(n)
            X = in_data

        # Add new centroids/data to existing clusters
        idxs, centroids = self._kshape(np.array(X), weights=weights)
        self._update_clusters(idxs, centroids, X, weights=weights)

        # Package into clusters
        clusters = []
        for i, centroid in enumerate(centroids):
            series = []
            for j, val in enumerate(idxs):
                if i == val:
                    series.append(j)
            clusters.append((centroid, series))

        # Append outliers at the end
        outliers = np.where(idxs==self.k)
        clusters.append(outliers)

        # Don't randomly initialize next batch
        self.init_random = False
        return clusters

    def _update_clusters(self, idxs, centroids, X, weights=None):
        """
        Update the clusters of this kshapestream with new data.
        idxs - n. cluster assignments of time series in X
        centroids - k x m array of centroids.
        X - n x m matrix of data
        weights - n. optional weights for each time series in X
        """
        n = X.shape[0]
        # If no weights, are given, we assume no weighting
        if weights is None:
            weights = np.ones(n)
        else:
            weights = np.array(weights)

        # Update cluster statistcs
        # Get distances of X to centroids
        distances = (1 - ncc_c_3dim(X, centroids).max(axis=2)).T
        # Update cluster statistics
        means, sigs, dists2 = self.update_cluster_stats(idxs, distances)
        self.means = means; self.sigs = sigs; self.dists2 = dists2

        # Iteratively update cluster shape matrices
        for i in range(self.k):
            centroid = centroids[i, :]
            # Update centroid
            self.centers[i, :] = centroid
            # Update shape matrix
            Xi = X[idxs==i, :] * weights[idxs==i, np.newaxis]
            self.S[i, :, :] = self.S[i, :, :] + np.dot(Xi.transpose(), Xi)
            # Update cluster counts
            self.counts[i] += np.sum(weights[idxs==i])

    """
    Probablistic Distance
    """
    def prob_idx(self, means, sigs, distances):
        """
        Assign each time series to "most likely" cluster given
        distance to clusters and existing distance statistics for each cluster
        means - k. mean of distance to centroid in each cluster
        sigs - k. std dev of distance to centroid in each cluster
        distances - n x k distances between each new time series and existing centroids
        returns:
        idxs - n. array of ints (0 to k-1) of cluster assignments
        """
        n = np.shape(distances)[0]

        # Convert distances to number of std devs from mean
        sig_dist = np.abs(distances-means)/sigs

        # Initialize cluster membership array
        idxs = np.zeros(n)
        for i in range(n):
            # Find the maximum likelihood cluster
            min_dist = np.argmin(sig_dist[i, :])
            if sig_dist[i, min_dist] < self.thresh:
                idxs[i] = min_dist
            else:
                # Outlier
                idxs[i] = self.k
        return idxs

    def update_cluster_stats(self, idxs, distances):
        """
        Update means, std devs of distances to centroid for each cluster for n new time series
        idxs - n. cluster assignments of time series
        distances - n x k distances between each new time series and existing centroids
        """
        # Start with previous means / variances / dist2 of clusters
        means = self.means.copy(); sigs = self.sigs.copy(); dists2 = self.dists2.copy()
        # Iterate through clusters and update stats
        for i in range(self.k):
            # Find total number of points in the cluster
            nprev = self.counts[i]; ncur = np.sum(idxs==i)
            ntot = nprev + ncur
            # If points have been added (in this dataset), update stats
            if ncur>0:
                d = distances[idxs==i, i]
                # Mean is the weighted average of previous and current mean
                means[i] = (means[i]*nprev + np.sum(d)) / ntot
                # Compute squared distance
                dists2[i] = (dists2[i]*nprev + np.sum(d**2)) / ntot
                # Sigmas are computed with a weighting, to increase
                # certainty with ntot
                sig = np.sqrt(dists2[i]-means[i]**2)
                weight = ntot / (1 + ntot)
                sigs[i] = weight*sig + (1-weight)*1
        return means, sigs , dists2

    """
    Visualization Methods
    """
    # This differs from that of the parent class because we show outliers
    def visualize(self, clusters, X, axes):
        """
        Visualize clusters
        clusters - list of (centroid, indices) for each cluster and list of outliers as last element
        X - n x m of clustered data
        cols - number of columns to plot
        """

        if len(axes) != self.k + 1:
            print('Error: Not enough axes')
            return

        colors = self.ps['centroid_colors']
        for i in range(self.k):
            ax = axes[i]
            centroid, els = clusters[i]
            if len(els) != 0:
                for j in els:
                    _, opt_x = sbd(centroid, X[j, :])
                    ax.plot(opt_x, self.ps['member_color'], alpha=self.ps['member_alpha'], linewidth=self.ps['member_size'])
            ax.plot(centroid, color=colors[i], linewidth=self.ps['centroid_size'])

            # Number of points in cluster
            ax.text(0.9, 0.9, str(int(self.counts[i])), horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes, fontsize=self.ps['count_size'])
            # Remove ticks
            if not self.ps['show_x']:
                ax.get_xaxis().set_ticks([])
            if not self.ps['show_y']:
                ax.get_yaxis().set_ticks([])
            ax.set_facecolor(self.ps['background_color'])

            # An inset showing distribution of distances in the cluster
            axin = inset_axes(ax, width="40%", height="30%", loc=4)
            mu = self.means[i]
            sig = self.sigs[i]
            gx = np.arange(-1, 1, 0.01)
            g = gaussian(gx, mu, sig)
            axin.plot(gx, g, color=self.ps['dist_color'])
            axin.fill_between(gx, g, alpha=self.ps['dist_alpha'], color=self.ps['dist_color'])
            axin.axis('off')

        # Plot outliers
        ax = axes[self.k]; ax.set_facecolor(self.ps['background_color']);
        if not self.ps['show_x']:
            ax.get_xaxis().set_ticks([])
        if not self.ps['show_y']:
            ax.get_yaxis().set_ticks([])
        els = clusters[self.k][0]
        if len(els) != 0:
            for j in els:
                ax.plot(X[j, :].T, linewidth=self.ps['member_size'], color=self.ps['member_color'], alpha=self.ps['member_alpha'])
            # Number of outliers
            ax.text(0.9, 0.9, '+' + str(int(len(els))), horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes, fontsize=self.ps['count_size'])