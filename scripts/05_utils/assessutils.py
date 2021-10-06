import numpy as np
from scipy import signal, stats

def compute_RMSD(y, y_pred):
    '''
    credit vineet et al from UCRB repo April 2021
    '''
    ngrid = y.size
    return np.sqrt(1 / ngrid * ((np.sum(y - y_pred))**2))


def compute_NSE(y, y_pred):
    '''
    credit vineet et al from UCRB repo April 2021
    '''
    return 1 - np.sum((y - y_pred)**2)/np.sum((y - np.mean(y))**2)


def compute_KGE(y, y_pred):
    '''
    credit vineet et al from UCRB repo April 2021
    '''
    r, _ = stats.pearsonr(y.flatten(), y_pred.flatten())
    a = np.std(y_pred.flatten()) / np.std(y.flatten())
    b = np.mean(y_pred.flatten()) / np.mean(y.flatten())
    return 1 - np.sqrt((r - 1)**2 + (a - 1)**2 + (b - 1)**2)

def compute_stats(y, y_pred):
    '''
    credit vineet et al from UCRB repo April 2021
    '''
    return np.array([compute_RMSD(y, y_pred), compute_NSE(y, y_pred), compute_KGE(y, y_pred)])