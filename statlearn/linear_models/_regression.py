import numpy as np
import pandas as pd
from scipy import stats


class LinearRegressor(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.slope = None
        self.intercept = None
        self.is_fit = False

    def fit(self, df, response):
        df = df.copy()
        df['intercept'] = 1
        x = df.drop(columns=response)
        variables = x.columns.to_list()
        x = x.values
        y = df[response].values

        # Least squares
        a = np.matmul(x.T, x)
        b = np.matmul(np.linalg.inv(a), x.T)
        beta = np.matmul(b, y)
        self.slope = beta[:-1]
        self.intercept = beta[-1]

        # Standard error
        y_hat = np.matmul(x, beta)
        residuals = y-y_hat
        rss = np.sum(residuals**2)# Residual sum of squares
        n = x.shape[0]# Number of samples
        k = x.shape[1]# Number of independent variables
        variance = rss/(n-k)
        cov = variance*np.linalg.inv(a)
        se = np.sqrt(np.diag(cov))# Standard error

        # t-statistic
        t = beta/se

        # p-value
        pval = stats.t.sf(np.abs(t), n-k)*2

        # Results
        res = pd.DataFrame({
            'coefficient': variables,
            'value': beta,
            'std error': se,
            't-statistic': t,
            'p-value': pval
        })

        self.is_fit = True

        return res