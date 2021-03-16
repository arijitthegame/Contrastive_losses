#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from sklearn.base import BaseEstimator
import pickle
import os
import gzip

#Assume the data points are flattened vectors. 
#If not, flatten them before using this code.
#Class labels are assumed to be integers and not one-hot 

#TODO Fix saving the model, add more kwargs

def symmetrize(W):
    """ Symmetrize matrix. """
    W[:] = 0.5 * (W + W.T)
    return W

#TODO deal with W having all negative eigenvalues
#WARNING: Grabage result if above happens 
def pos_semidefinite(W):
    """ Make matrix positive semi-definite. """
    w, v = np.linalg.eig(symmetrize(W))  # eigvec in columns, real eigenvalues
    D = np.diagflat(np.maximum(w, 0))
    W[:] = np.dot(np.dot(v, D), v.T)
    return W




class KernelTriplet(BaseEstimator):
    """ OASIS algorithm. """

    def __init__(self, aggress=0.1, symmetrize=False,
                 make_pos_sd=False, n_iter=10 ** 4, sym_every=1,
                 psd_every=1, random_seed=42, save_every= None, save_path=None):
# If you want to visualize the kernel, set make_pos_sd = True
        
        self.aggress = aggress
        self.n_iter = n_iter
        self.symmetrize = symmetrize
        self.make_pos_sd = make_pos_sd
        self.sym_every = sym_every
        self.psd_every = psd_every
        self.random_seed = random_seed #For reproducibility
        self.save_every = save_every
        self.save_path = save_path
        
        if save_every is None:
            self.save_every = int(np.ceil(self.n_iter / 10))
        else:
            self.save_every = save_every

        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)


#Helper functions for saving and loading weights
    def _getstate(self):
        return (self._weights, )

    def _setstate(self, state):
        weights, = state
        self._weights = weights
        
    def _save(self, n=None):
        """ Pickle the model."""
        fname = self.save_path + "/model%04d" % n
        f = gzip.open(fname, 'wb')
        state = self._getstate()
        pickle.dump(state, f)
        f.close()

    def load_model(self, fname):
        """ Load model state from pickle. """
        f = gzip.open(fname, 'rb')
        state = pickle.load(f)
        self._setstate(state)

    def fit_batch(self, W, X, y, class_start, class_sizes, n_iter, margin=1.0):
        """ Train batch inner loop. """

        loss_steps_batch = np.empty((n_iter,), dtype='bool')
        n_samples, n_features = X.shape

        assert(W.shape[0] == n_features)
        assert(W.shape[1] == n_features)

        for i in range(n_iter):

            # Sample an anchor index
            anchor_index = self.init.randint(n_samples)
            label = y[anchor_index]

            # Draw random positive sample index
            pos_index = class_start[label] + \
                self.init.randint(class_sizes[label])

            # Draw random negative sample
            neg_index = self.init.randint(n_samples)
            while y[neg_index] == label:
                neg_index = self.init.randint(n_samples)

            anchor = X[anchor_index]

            sample_dist = X[pos_index] - X[neg_index]

            loss = margin - np.dot(np.dot(anchor, W), sample_dist)

            if loss > 0:
                # Update W
                grad_W = np.outer(anchor, sample_dist)

                loss_steps_batch[i] = True

                norm_grad_W = np.dot(anchor, anchor) * np.dot(sample_dist,
                                                    sample_dist)

                # constraint on the maximal update step size
                tau_val = loss / norm_grad_W  
                tau = np.minimum(self.aggress, tau_val)

                W += tau * grad_W
                

        return W, loss_steps_batch
    
    def fit(self, X, y):
        """ Fit the model """
        
        n_samples, n_features = X.shape

        self.init = np.random.RandomState(self.random_seed)
        
       # self._weights = np.eye(n_features).flatten()
        self._weights = np.eye(n_features)
        
        # self._weights = np.random.randn(n_features,n_features).flatten()
      #  W = self._weights.view()
        W = self._weights
      #  W.shape = (n_features, n_features)
        
        indices = np.argsort(y)

        y = y[indices]
        X = X[indices, :]

        classes = np.unique(y)
        classes.sort()

        n_classes = len(classes)

        # Translate class labels to serial integers 0, 1, ...
        y_new = np.empty((n_samples,), dtype='int')
        
        for i in range(n_classes):
            y_new[y == classes[i]] = i

        y = y_new
        class_sizes = [None] * n_classes
        class_start = [None] * n_classes

        for i in range(n_classes):
            class_sizes[i] = np.sum(y == i)
            # This finds the first occurrence of that class
            class_start[i] = np.flatnonzero(y == i)[0]

        loss_steps = np.empty((self.n_iter,), dtype='bool')
        n_batches = int(np.ceil(self.n_iter / self.save_every))
        steps_vec = np.ones((n_batches,), dtype='int') * self.save_every
        steps_vec[-1] = self.n_iter - (n_batches - 1) * self.save_every


        for b in range(n_batches):

            W, loss_steps_batch = self.fit_batch(W, X, y,
                                                  class_start,
                                                  class_sizes,
                                                  steps_vec[b],
                                                  1.0)
            
            loss_steps[b * self.save_every:min(
                (b + 1) * self.save_every, self.n_iter)] = loss_steps_batch

            if self.symmetrize:
                if np.mod(b + 1, self.sym_every) == 0 or b == n_batches - 1:
                    symmetrize(W)

            if self.make_pos_sd:
                if np.mod(b + 1, self.psd_every) == 0 or b == n_batches - 1:
                    pos_semidefinite(W)

    

        return self
    
    def predict(self, X_test, X_train, y_test, y_train, maxk=200):
        '''
        Evaluate the model by KNN classification. 

        '''
    # Maxk needs to be smaller than the training set. 
      #  W = self._weights.view()
        W = self._weights
      #  W.shape = (int(np.sqrt(W.shape[0])), int(np.sqrt(W.shape[0])))
        assert(W.shape[0] == X_test.shape[1])
        assert(W.shape[1] == X_test.shape[1])

        maxk = min(maxk, X_train.shape[0])  # K can't be > numcases in X_train

        numqueries = X_test.shape[0]

        precomp = np.dot(W, X_train.T)

        # compute similarity scores
        s = np.dot(X_test, precomp)

        # argsort sorts in ascending order
        # so we need to reverse the second dimension
        ind = np.argsort(s, axis=1)[:, ::-1]

        # Voting based on nearest neighbours
        # make sure it is int
        # We do not want KNN classification but just save the s matrix
        #TODO Add a save path to the matrix s

       
        if y_train.dtype.kind != 'int':
            queryvotes = y_train[ind[:, :maxk]].astype('int')
        else:
            queryvotes = y_train[ind[:, :maxk]]

        errsum = np.empty((maxk,))

        for k in range(maxk):
            # AFAIK bincount only works on vectors
            # so we must loop here over data items
            labels = np.empty((numqueries,), dtype='int')
            for i in range(numqueries):
                b = np.bincount(queryvotes[i, :k + 1])
                labels[i] = np.argmax(b)  # get winning class

            errors = labels != y_test
            errsum[k] = sum(errors)

        errrate = errsum / numqueries
        return errrate

    
    
 
        

        
        
        
        
        