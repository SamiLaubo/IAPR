import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import binascii
import struct
import numpy as np
import scipy
import scipy.misc
import scipy.cluster

from colorthief import ColorThief

from scipy import ndimage as ndi

from skimage.filters import gabor_kernel


def histogramFeatureExtraction(puzzlePiece, bins = 256, plotHist = False):
    '''
    Takes a single puzzle piece as input and returns a merged vector of the red, green and blue histogram.

    Parameters: 
        puzzlePiece: Single [128,128,3] picture of puzzle piece
        bins: Number of bins in the histogram
        plotHist: Plots puzzle piece along with its three histograms
    
    Returns:
        ndarray of size 3*bins
    '''

    ## Code taken and modified from: https://www.pinecone.io/learn/color-histograms/
    
    # Extract histogram for all tree colors using cv2.Hist

    red_hist = cv2.calcHist( ### NB! Im not 100% sure about which is red and which is blue, cv2 is weird
        [puzzlePiece], [0], None, [bins], [0, 256]
        #[images], [channels], [mask], [bins], [hist_range]
    )
    green_hist = cv2.calcHist(
        [puzzlePiece], [1], None, [bins], [0, 256]
    )
    blue_hist = cv2.calcHist(
        [puzzlePiece], [2], None, [bins], [0, 256]
    )

    # Gather the three histograms into a single one
    histVector = np.concatenate([red_hist, green_hist, blue_hist], axis=0)
    # Reshape the array
    histVector = histVector.reshape(-1)

    # Plot histograms along with a picture of puzzle if true
    if plotHist:
        plt.imshow(puzzlePiece)
        plt.show()

        fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        axs[0].plot(red_hist, color='r')
        axs[1].plot(green_hist, color='g')
        axs[2].plot(blue_hist, color='b')
        plt.show()

    return histVector # Return the histogram array


### Gabor starts here

def compute_feats(image, kernels, chanels = 3, flatten = True):
    '''
    Computes a feature vector for using different Gabor filters.

    Parameters:
        image: Single image of a tile
        kernels: List of Gabor filters used to convolve the image
        chanels: Number of chanels in the image, 3 if RGB
        flatten: Return the image as a one dimentional array
    Returns:
        Array of features

    '''

    totalFeats = [] # List of all features from all channels

    for i in range(chanels): # Iterate through channels

        feats = np.zeros((len(kernels), 2), dtype=np.double) # Features for single channel

        for k, kernel in enumerate(kernels): # Iterate through kernels

            filtered = ndi.convolve(image[:,:,i], kernel, mode='wrap') # Convolved image
            feats[k, 0] = filtered.mean() # Mean of convolved image
            feats[k, 1] = filtered.var() # Variance of convolved image

        totalFeats.append(feats) # Append these features to the total

    if flatten:
        return np.concatenate(totalFeats).ravel()
    else:
        return totalFeats


# prepare filter bank kernels