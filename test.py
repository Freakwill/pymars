#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Predict with Mars in R/rpy2
"""

import mars

import pandas as pd
import numpy as np
import numpy.linalg as LA

x1 = np.linspace(-1,1,10)
x2 = np.linspace(-1,1,10)
x2, x1=np.meshgrid(x2,x1)
y1 = np.abs(x1) + np.abs(x2)
y2= np.abs(x1) + np.abs(x2)*np.abs(x1)

Xtrain = np.column_stack([x1.flatten(), x2.flatten()])
Ytrain=np.column_stack([y1.flatten(), y2.flatten()])
model = mars.Mars(degree=2, thresh=0.0000001)
model.fit(Xtrain, Ytrain)

Ypred = model.predict(Xtrain)

print(LA.norm(Ypred-Ytrain)/LA.norm(Ytrain))
