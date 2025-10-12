# -*- Encoding: Latin-1 -*-
#!/usr/bin/python

from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import det, svd
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import kron, coo_matrix


# ------------------------------------------------------------------------------
# 1: PROJECTION ALGORITHMS
# ------------------------------------------------------------------------------

def projection(Y, n_sources, method='pca'):

    """
    Performs the projection of the dataset on the reduced dimension space
    The projection can be performed by relying on a PCA or on a SVD transform
    
    Ref: Bioucas Dias et al. 
   
    Parameters
    ----------
        
    Y: numpy array (size: n_bands by n_samples)
      data matrix
    n_sources: int
      number of endmembers. The number of endmembers determines the dimension
      of the reduced dimension space.
    method: string (default: 'pca')
      method used to perform the projection ('pca' or 'svd')
      
    Return
    ------
    
    Y_rec: numpy array (size: n_bands by n_samples)
      data reconstruction from the projected data after applying an inverse
      transform
    Y_proj: numpy array
      projected data
    """
    
    if(method == None):
    
        SNR = estimate_SNR(Y, Y_rec, n_sources)
        SNR_th = 15 + 10*np.log10(n_sources)
        if(SNR < SNR_th):
            print('Selection of the PCA projection')
            method = 'pca'
        else:
            print('SVD projection')
            method = 'svd'
        
    if(method == 'pca'):
    
        pca = PCA(n_components=n_sources - 1)
        Y_proj = pca.fit_transform(Y.T)
        Y_rec = pca.inverse_transform(Y_proj)
        
    elif(method == 'svd'):
        
        svd = TruncatedSVD(n_components=n_sources)
        Y_proj = svd.fit_transform(Y.T).T
        u = np.mean(Y_proj, axis=1).reshape((1, -1))
        Y_proj /= np.dot(u, Y_proj)
        Y_rec = svd.inverse_transform(Y_proj.T).T
        
    return Y_proj, Y_rec
    
   
# ------------------------------------------------------------------------------
# ENDMEMBERS ESTIMATION: Pure-pixel based algorithms
# ------------------------------------------------------------------------------        
        
class NFINDR:

    """
    N-FINDR algorithm implementation
        
    Ref: Winter, M. E. (1999). N-FINDR: An algorithm for fast autonomous 
    spectral endmember determination in hyperspectral data. In: Imaging 
    Spectrometry V (Vol. 3753, pp. 266-275). International Society for Optics 
    and Photonics.
    
    Parameters
    ----------
        
    Y: numpy array (size: n_bands by n_samples)
      data matrix
    n_sources: int
      number of endmembers
          
    Attributes
    ----------

    Y: numpy array (size: n_bands by n_samples)
      data points
    n_samples: int
      number of samples in the dataset
    n_sources: int
      number of endmembers
    n_bands: int
      spectral dimension
    Y_proj: numpy array (size: n_sample by n_sources-1 matrix)
      data points in the reduced dimension space 
    M_proj: numpy array (size: n_sources by n_sources-1 matrix)
      endmembers vectors in the reduced dimension space 
    vol: float
      volume of the simplex formed by the endmembers
    M: numpy array (size: n_bands by n_sources)
      endmembers matrix 
    """

    def __init__(self, Y, n_sources):

        self.Y = Y
        self.n_bands, self.n_samples = self.Y.shape
        self.n_sources = n_sources

        # Project the data points
        pca = PCA(n_components=self.n_sources - 1)
        self.Y_proj = pca.fit_transform(Y.T)
        
        # Select random data points as initial guess for the endmembers
        random_indices = np.random.choice(self.n_samples, size=self.n_sources, 
          replace=False)
        self.M_proj = self.Y_proj[random_indices, :]

        # Computes the volume of the endmembers simplex
        self.vol = abs(det(self.M_proj[1:, :] - self.M_proj[0, :]))


    def run(self):

        """
        Run the N-FINDR algorithm
        """
        indexes = []
        for s in range(self.n_sources):

            endmembers = np.copy(self.M_proj)
            idx = 0

            # Iterate over the data points
            for n in range(self.n_samples):

                # Try replacing the selected endmember by the data point
                endmembers[s, :] = self.Y_proj[n, :]
                vol = abs(det(endmembers[1:, :] - endmembers[0, :]))

                # Update the endmember if the volume is greater than 
                # the current one
                if(vol > self.vol):
                    self.M_proj = np.copy(endmembers)
                    self.vol = vol
                    idx = n

            indexes.append(idx)

        self.M = self.Y[:, np.array(indexes)]
        
  
    def display(self):

        """
        Display the data points in the reduced dimension space. 
        This visualization method is mostly appropriate when the number of 
        sources equals 3.
        """
        
        if(self.n_sources != 3):
            print("Warning: the number of sources is larger than 3")

        plt.figure()
        plt.scatter(self.Y_proj[:, 0], self.Y_proj[:, 1], color='blue')
        plt.scatter(self.M_proj[:, 0], self.M_proj[:, 1], color='yellow')
        plt.show()
        
