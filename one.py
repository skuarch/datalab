import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

def ReturnDataFrame(path):
        return pd.read_csv(path, sep=',',skipinitialspace=True)
        

# Load CVS
Path1 = './voice.csv'
DataMatrix = ReturnDataFrame(Path1)

DataMatrix.replace({'male': -1.0, 'female': 1.0},
                    inplace=True)

DataLabels = DataMatrix['label']

DataMatrix.drop('label', axis=1, inplace=True)

# Transform to an NP Array
Data = DataMatrix.as_matrix()
Label = DataLabels.as_matrix()

np.mean(Data,axis = 0)

Data = np.matrix(Data)
DataMean = Data - np.mean(Data,axis = 0)

Cov = DataMean.T*DataMean

n1, n2 = Data.shape

Cov = (1/float(n1))*Cov

Eigenvaluesc, Eigenvectorsc = np.linalg.eigh(Cov) 
idx = Eigenvaluesc.argsort()[::-1]  
Eigenvaluesc = Eigenvaluesc[idx]
Eigenvectorsc  =  Eigenvectorsc [:,idx]


plt.ion()
plt.plot(Eigenvaluesc, Eigenvaluesc, 'ro')
plt.axis([0, 6, 0, 20])
plt.show()
