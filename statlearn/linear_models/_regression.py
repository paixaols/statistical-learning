import numpy as np
import pandas as pd
from scipy import stats


class LinearRegressor(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.slope = None
        self.intercept = None
        self.coef = None
        self.stats = None
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

        # Accuracy of the model
        y_hat = np.matmul(x, beta)
        residuals = y-y_hat
        rss = np.sum(residuals**2)# Residual sum of squares

        n = x.shape[0]# Number of samples
        k = x.shape[1]# Number of independent variables + 1
        variance = rss/(n-k)
        cov = variance*np.linalg.inv(a)
        se = np.sqrt(np.diag(cov))# Standard error

        y_bar = np.mean(y)
        tss = np.sum((y-y_bar)**2)# Total sum of squares
        R2 = 1-rss/tss
        rse = np.sqrt(rss/(n-k))# Residual standard error

        # t-statistic
        t = beta/se

        # p-value
        pval = stats.t.sf(np.abs(t), n-k)*2

        # Results
        self.coef = pd.DataFrame({
            'coefficient': variables,
            'value': beta,
            'std error': se,
            't-statistic': t,
            'p-value': pval
        })
        self.stats = pd.DataFrame({
            'RSS': rss,
            'TSS': tss,
            'RSE': rse,
            'R^2': R2
        }, index=[0]).T.reset_index()
        self.stats.columns = ['statistic', 'value']

        self.is_fit = True

        return self.coef