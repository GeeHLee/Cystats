from Cystats import OLS, ADF
import numpy as np

import statsmodels.api as sm

X = 2 * np.random.rand(100,1) 
adf = ADF()
adf.add_const(X, False)

X = sm.add_constant(X)
Y = np.dot(np.array([2, 3]), X.T) + 5
Y = Y.reshape(-1, )

stat = sm.OLS(Y, X).fit()
model = OLS()
model.fit(X, Y)
model.params
model.aic
stat.aic







