#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Code accompanying the manuscript:
"A new link-species relationship connects ecosystem structure and stability"

-------
v. 0.1 ; August 2020
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
    Computes the average number of extinctions occurring after the removal of one species.
    Returns the coefficient of determination of its prediction based on Equation 6.
    
    Parameters
    ----------
    sseq : numpy array of shape (nbsimu, S) with nbsimu being the number of network decompositions done.
        Each row of this array contains the species richness along one decomposition.
    
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
        of one species (Equation 6).
    
    r2lost : float
        Coefficient of determination for the predictions of 
        the number of species lost based on the number of species removed.
        
    lost : numpy array of shape (nbsimu, S)
        Number of species lost along each decomposition.
        
    removed : numpy array of shape (nbsimu, S)
        Number of species removed along each decomposition.
    
    """
    #### Number of species removed and lost ####
    nbsimu, S = sseq.shape
    removed = np.array(nbsimu*[np.arange(S)]) # Number of species removed
    lost = S-sseq # Number of species lost

    #### Predicted slope of lost~removed (deltaS, Equation 6) ####
    deltaS = 1 + (1-z)*((2/b)-1)
    
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
        Mean robustness over nbsimu network decompositions
        
    var : float
        Observed variance of robustness over nbsimu network decompositions
        
    pred : float
        Predicted robustness based on Equation 7.
   
    """     
    # Fraction of species lost
    lost = 1-(sseq/S)
    # Robustness
    robs = np.sum(lost < x, axis=1)/S
    # Prediction (Equation 7)
    pred = np.ceil(x*S)/(
                    S*((1-z)*(2/b)+z))
    res = namedtuple('Robres', ('obs', 'var', 'pred'))
    return(res(np.mean(robs), np.var(robs), pred))
