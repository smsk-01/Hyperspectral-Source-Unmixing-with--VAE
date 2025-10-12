import utils
import os
import numpy as np
import matplotlib.pyplot as plt
from algorithms import *


hsi,_,_ = utils.load_HSI("./dataset/Samson.mat") # load Samson Dataset
data = hsi.get_spectra() # Get the hyperspectral observation 
A = hsi.get_abundances() # Get the ground truth for the abundances
E = hsi.endmembers # Get the ground truth for the endmembers
n_sources = 3

# Display the hyperspectral observations in the reduced dimension space
Yproj, _ = projection(data.T, n_sources, method='pca')
plt.scatter(Yproj[:, 0], Yproj[:, 1], color='blue')
plt.show()

# Endmember estimation with NFINDR algorithm
nfindr = NFINDR(data.T, n_sources)
nfindr.run()
nfindr.display()

plt.figure()
plt.plot(E[0], '--', color='green', label='Groundtruth')
plt.plot(E[1], '--', color='green')
plt.plot(E[2], '--', color='green')
plt.plot(nfindr.M[:, 0], color='green', label='Estimation')
plt.plot(nfindr.M[:, 1], color='green')
plt.plot(nfindr.M[:, 2], color='green')
plt.legend()
plt.show()


