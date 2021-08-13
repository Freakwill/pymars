#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Predict with Mars in R/rpy2
MARS: Multivariate Adaptive Regression Splines
"""

import rpy2
import rpy2.robjects as ro
import rpy2.rinterface as ri
import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
r = ro.r

# _ = importr('class')
mda = importr('mda')

from sklearn.base import RegressorMixin


class Mars(RegressorMixin):
    '''MARS: Multivariate Adaptive Regression Splines
    
    MARS is an adaptive procedure for regression, and is well suited for high-dim problems.
    It uses expansions in basis functions {(x-t)+, (x-t)-, ...}
    
    Extends:
        RegressorMixin
    
    Variables:
        xkeys {Array} -- [description]
        ykeys {Array} -- [description]
    '''

    xkeys = None
    ykeys = None
    def __init__(self, degree=1, nk=None, penalty=2, thresh=0.001, prune=True, trace_mars=False,
         forward_step=True, prevfit=False):
        self.degree=degree
        self.nk = nk
        self.penalty = penalty
        self.thresh = thresh
        self.prune = prune
        self.trace_mars = trace_mars
        self.forward_step = forward_step
        self.prevfit = prevfit

    @property
    def estimator(self):
        return self.__estimator

    @estimator.setter
    def estimator(self, x):
        self.__estimator = x
    

    def get_params(self, deep=True):
        return {'degree':self.degree, 'nk':self.nk, 'penalty':self.penalty, 'trace_mars':self.trace_mars, 'forward_step':self.forward_step, 'prevfit':self.prevfit}

    def fit(self, X, Y, sample_weight=None):
        self.outdim = Y.ndim
        if isinstance(X, pd.DataFrame):
            self.xkeys = X.columns
            Xtrain = pandas2ri.py2rpy_pandasdataframe(X)
        else:
            if self.xkeys is None:
                self.xkeys = ['x%d'%k for k in range(X.shape[1])]
            Xtrain=ro.DataFrame({key:ro.FloatVector(X[:,k].astype(np.float64, copy=False)) for k, key in enumerate(self.xkeys)})
        if self.ykeys is None:
            if self.outdim == 1:
                self.ykeys = ['y1']
            else:
                self.ykeys = ['y%d'%k for k in range(Y.shape[1])]
        if self.outdim == 1:
            Ytrain=ro.DataFrame({self.ykeys[0]:ro.FloatVector(Y)})
        else:
            Ytrain=ro.DataFrame({key:ro.FloatVector(Y[:,k].astype(np.float64, copy=False)) for k, key in enumerate(self.ykeys)})
        
        if self.nk is None:
            self.nk = max((21, 2*X.shape[1]+1))
        if sample_weight is None:
            self.estimator = mda.mars(Xtrain, Ytrain, degree=self.degree, nk=self.nk, penalty=self.penalty, trace_mars=self.trace_mars, forward_step=self.forward_step, prevfit=self.prevfit)
        else:
            self.estimator = mda.mars(Xtrain, Ytrain, w=ro.FloatVector(sample_weight), degree=self.degree, nk=self.nk, penalty=self.penalty, trace_mars=self.trace_mars, forward_step=self.forward_step, prevfit=self.prevfit)
        return self

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        Xtest=ro.DataFrame({key:ro.FloatVector(X[:,k].astype(np.float64, copy=False)) for k, key in enumerate(self.xkeys)})
        Ypred = r.predict(self.estimator, Xtest)
        if self.outdim==1:
            return np.array(Ypred)[:,0]
        else:
            return np.array(Ypred)
