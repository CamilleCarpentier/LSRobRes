#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code accompanying the manuscript:
"Reinterpreting the relationship between number of species and 
number of links connects community structure and stability"

-------
v1.0.0 (First release)
-------

For any question or comment, please contact:
Camille Carpentier(1), camille.carpentier@unamur.be 

(1) University of Namur, Namur, Rue de Bruxelles 61, Belgium.
Research Unit in Environmental and Evolutionary Biology (URBE);
Institute of Life-Earth-Environment, Namur; 
Center for Complex Systems, Namur.

"""
import numpy as np
from collections import namedtuple

##############################################################################
# Average number of extinctions occurring after the removal of one species
##############################################################################

def R2deltaS(sseq, S, b, z):
    """
    Computes the average number of extinctions occurring after the removal of 
    one species.
    Returns the coefficient of determination of its prediction.
    The prediction is based on Equation 4.
    
    Parameters
    ----------
    sseq : numpy array of shape (nbsimu, S) with nbsimu being the number of 
        network decompositions done.
        Each row of this array contains the species richness along one 
        decomposition.
    
    S : int
        Number of species
    
    b : float
        Shape of the link~species relationship given by b = log(L)/log(S/2)

    z : float
        Proportion of species which cannot undergo secondary extinction
        
    Returns
    -------
    
    deltaS : float
        Predicted average number of extinctions occurring after the removal 
        of one species (Equation 4).
    
    r2lost : float
        Coefficient of determination for the predictions of 
        the number of species lost based on the number of species removed.
        
    lost : numpy array of shape (nbsimu, S)
        Number of species lost along each decomposition.
        
    removed : numpy array of shape (nbsimu, S)
        Number of species removed along each decomposition.
    
    """
    #### Number of species removed and lost ####
    nbsimu = sseq.shape[0] # Each row is one decomposition
    removed = np.array(nbsimu*[np.arange(sseq.shape[1])]) # Number of species removed
    lost = S-sseq # Number of species lost

    #### Predicted slope of lost~removed (deltaS) ####
    deltaS = (1-z)*(2/b)+z # Equation 4
    
    #### Coefficient of determination ####
    obs = lost # Observed values
    pred = deltaS*removed # Predicted values
    SSres = np.sum((obs-pred)**2) # Residual sum of squares
    SStot = np.sum((obs-np.mean(obs))**2) # Total sum of squares
    r2lost = 1 - (SSres/SStot) # Coefficient of determination
    
    result = namedtuple('R2deltaSres', ('deltaS', 'r2lost', 'lost', 'removed'))
    return(result(deltaS, r2lost, lost, removed))
  
##############################################################################
# Robustness
##############################################################################
    

def robx(sseq, S, b, z, x = 0.5):  
    """
    Returns observed and predicted robustness at threshold x.
    
    Parameters
    ----------
    
    sseq : numpy array of shape (nbsimu, S) with nbsimu being the number of network decompositions done.
        Each row of this array contains the species richness along one decomposition sequence.
    
    S : int
        Number of species.
    
    b : float
        Shape of the links~species relationship given by b = log(L)/log(S/2).
    
    z : float
        Proportion of species which cannot undergo secondary extinction.
    
    x : float
        Robustness threshold (fraction of species lost to reach).
    
    Returns
    -------
    obs : float
        Mean robustness over nbsimu network decompositions.
        
    var : float
        Observed variance of robustness over nbsimu network decompositions.
        
    pred : float
        Predicted robustness based on Equation 5.
   
    """     
    # Fraction of species lost
    lost = 1-(sseq/S)
    # Robustness
    robs = np.sum(lost < x, axis=1)/S
    # Prediction (Equation 5 + cf. Methods for the ceil(.))
    pred = np.ceil(x*S)/(
                    S*((1-z)*(2/b)+z))
    res = namedtuple('Robres', ('obs', 'var', 'pred'))
    return(res(np.mean(robs), np.var(robs), pred))
