from Cystats import OLS, ADF
import numpy as np
import unittest

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, lagmat, mackinnonp

class TestOLS(unittest.TestCase):
    
    def setUp(self):
        X = 2 * np.random.rand(100, 3)
        self.X = sm.add_constant(X)
        self.Y = np.dot(np.array([1, 2, 3]), X.T) + 5
        self.cppmodel = OLS()
        self.cppmodel.fit(self.X, self.Y, True)
        self.statmodel = sm.OLS(self.Y, self.X).fit()
    
    def test_params(self):
        self.assertTrue(np.allclose(self.cppmodel.params, self.statmodel.params))
    
    def test_predict(self):
        self.assertTrue(np.allclose(self.cppmodel.fittedvalues, self.statmodel.fittedvalues))
    
    def test_residual(self):
        self.assertTrue(np.allclose(self.cppmodel.resid, self.statmodel.resid))
    
    def test_stand_error(self):
        self.assertTrue(np.allclose(self.cppmodel.se, self.statmodel.bse))
        
class TestADF(unittest.TestCase):
    
    def setUp(self):
        self.X = np.random.rand(1, 100)[0]
        self.adf = ADF()
        self.ols = OLS()
        
    def test_lagmat(self):
        max_lag = self.adf.get_maxlag(self.X)
        self.assertTrue(np.allclose(self.adf.lagmat(self.X, max_lag), lagmat(self.X, max_lag, trim="both", original="in")))
    
    def test_stats(self):
        self.adf.run(self.X, self.ols)
        val = adfuller(self.X)
        self.assertAlmostEqual(self.adf.stat, val[0])
        
        pvalue = mackinnonp(self.adf.stat)
        self.assertAlmostEqual(pvalue, val[1])

if __name__ == '__main__':
    unittest.main()
