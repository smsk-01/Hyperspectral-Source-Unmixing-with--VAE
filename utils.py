# -*- Encoding: Latin-1 -*-
#!/usr/bin/python


from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ------------------------------------------------------------------------------
# Hyperspectral Image (HSI) data
# ------------------------------------------------------------------------------ 
 
        
class HSI:

    """
    Class used to represent Hyperspectral Image (HSI) data
    
    Parameters
    ----------
    
    data: numpy array
      spectral observations
    rows: int
      number of rows in the image
    cols: int
      number of columns in the image
    
    Attributes
    ----------
    
    data: numpy array 
      spectral observations
    rows: int
      number of rows in the image
    cols: int
      number of columns in the image
    bands: int
      number of spectral bands
    image: array-like  
      hyperspectral image
    gt: numpy array
      endmembers
    abundances_map: numpy array
      abundances map
    """
    
    def __init__(self, data, rows, cols, endmembers, abundances_map):
    
        if data.shape[0] < data.shape[1]:
            data = data.transpose()
            
        self.bands = data.shape[1]
        self.cols = cols
        self.rows = rows
        self.image = np.reshape(data,(self.rows, self.cols, self.bands))
       
        self.endmembers = endmembers
        self.abundances_map = abundances_map
        
        
    def get_spectra(self):
    
        """
        Returns the spectral observations
        
        Returns
        -------
        
        out: numpy array
          spectral observations
        """
    
        return np.reshape(self.image, (self.rows*self.cols, self.bands))
        
        
    def get_abundances(self):
    
        """
        Return the abundances associated with the spectral observations
        
        Return
        ------
        
        out: numpy array (size: n by k)
          abundances
        """
        return np.reshape(self.abundances_map, (self.rows*self.cols, -1))
    
    def get_bands(self, bands):
    
        """
        Return the channel of the HSI corresponding to the specified band
        
        Parameters
        ----------
        
        bands: int
          index of the spectral band to return
          
        Return
        ------
        
        out: numpy array
          channel of the HSI associated with the specified band
        """
        return self.image[:, :, bands]


    def crop_image(self,start_x,start_y,delta_x=None,delta_y=None):
    
        """
        Apply a spatial crop to the HSI and returns the cropped image
        
        Parameters
        ----------
        
        start_x, start_y: int
          coordinates of the top left pixel of the cropped image
        delta_x, delta_y: int
          shape of the cropped image
          if set to None, then the bottom right pixel of the cropped is taken
          to be the bottom right pixel of the original image
        
        Returns
        -------
        
        out: numpy array 
          cropped image
        """
        
        if delta_x is None: delta_x = self.cols - start_x
        if delta_y is None: delta_y = self.rows - start_y
        return self.image[start_x:delta_x+start_x,start_y:delta_y+start_y,:]


def load_HSI(path):

    """
    Load an hyperspectral image from a .mat file
    
    Parameters
    ----------
    
    path: string
      location of the .mat file
      
    Returns
    -------
    
    out: instance of the class HSI
      HSI data
    """
    
    data = loadmat(path)
    Y = np.asarray(data['Y'], dtype=np.float32)
    n_rows = data['lines'].item()
    n_cols = data['cols'].item()
    abundances_map = data['S_GT']
    
    if 'GT' in data.keys():
        gt = np.asarray(data['GT'], dtype=np.float32)
    else:
        gt = None
    
    return HSI(Y, n_rows, n_cols, gt, data['S_GT']), Y, data




