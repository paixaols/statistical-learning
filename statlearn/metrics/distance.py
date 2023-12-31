import numpy as np


def array_point(x, p, metric='euclidean'):
    '''
    Compute the distance between the elements of an array and a given point.

    Parameters
    ----------
        x: array-like of shape (n_samples, n_features).
        p: array-like of shape (n_features,).

    Returns
    -------
        Array of shape (n_samples,) with the distances between p and each point
        in x.
    '''
    x = np.array(x)
    p = np.array(p)
    if x.shape[1] != p.shape[0]:
        raise Exception(f'x and p are of shape {x.shape} and {p.shape}, should be (n, m) and (m,)')

#     TODO: Implementar outras métricas
    if metric == 'manhattan':
        pass
    else:# Usar métrica euclideana como padrão
        n_rows, n_cols = x.shape
        d = [0]*n_rows
        for i in range(n_rows):
            for j in range(n_cols):
                d[i] += (x[i][j]-p[j])**2
        return np.sqrt(d)
