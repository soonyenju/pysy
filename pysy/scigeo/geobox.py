from scipy.spatial import cKDTree as KDTree
import numpy as np

class IDW(object):
	""" 
	# https://mail.python.org/pipermail/scipy-user/2010-June/025920.html
	# https://github.com/soonyenju/pysy/blob/master/pysy/scigeo.py
	inverse-distance-weighted interpolation using KDTree:
	invdisttree = Invdisttree(X, z)  
	-- points, values
	interpol = invdisttree(q, k=6, eps=0)
	-- interpolate z from the 6 points nearest each q;
		q may be one point, or a batch of points

	"""
	def __init__(self, X, z, leafsize = 10):
		super()
		self.tree = KDTree(X, leafsize=leafsize)  # build the tree
		self.z = z

	def __call__(self, q, k = 8, eps = 0):
		# q is coor pairs like [[lon1, lat1], [lon2, lat2], [lon3, lat3]]
		# k nearest neighbours of each query point --
		# format q if only 1d coor pair passed like [lon1, lat1]
		if not isinstance(q, np.ndarray):
			q = np.array(q)
		if q.ndim == 1:
			q = q[np.newaxis, :]

		self.distances, self.ix = self.tree.query(q, k = k,eps = eps)
		interpol = []  # np.zeros((len(self.distances),) +np.shape(z[0]))
		for dist, ix in zip(self.distances, self.ix):
			if dist[0] > 1e-10:
				w = 1 / dist
				wz = np.dot(w, self.z[ix]) / np.sum(w)  # weightz s by 1/dist
			else:
				wz = self.z[ix[0]]
			interpol.append(wz)
		return interpol