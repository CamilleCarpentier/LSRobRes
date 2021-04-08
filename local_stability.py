#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Code accompanying the manuscript:
"Reinterpreting the relationship between number of species and 
number of links connects community structure and stability"

-------
v. 0.2 ; March 2021
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
# Main function
##############################################################################

def realpart(mat, inttype, mu = 0, sigma2 = 1, d = 0, nbsimu = 1000):
    """ 
    Computes observed (mean and variance over nbsimu matrices) and 
    predicted real part of the rightmost eigenvalue for community matrices.
    
    Parameters
    ----------
    mat : numpy array with shape (S, S), S being the number of species
        Adjacency matrix of the network (A)
        aij = 1; aji = 1 if two species interact
        aij = 0; aji = 0 if two species do not interact
    
    inttype : string
        Type of the interactions (either 'trophic' or 'mutualistic')
        If 'trophic', the aij and aji elements will be set by the function to 1 and -1.
    
    mu : float
        Mean interaction strength (0 by default).
    
    sigma2 : float
        Variance of interaction strength (1 by default).
    
    d : float or list
        Species self-regulation (diagonal elements of M = -d; 0 by default).
        
    
    Returns
    -------
    obs : float
        Mean of the observed real part of the  rightmost eigenvalues over nbsimu matrices.
        
    var : float
        Variance of the observed real part of the rightmost eigenvalues over nbsimu matrices. 
        
    pred : float
        Predicted real part of the rightmost eigenvalue based on Equation 7-8.
        
    """

    #### Random community matrices ####
    
    A = mat!=0 # Adjacency matrix A
    W = np.random.normal(mu,sigma2,
                         size=(nbsimu, *A.shape)) # Interaction strength matrix
    M = W*A[np.newaxis,:,:] # Community matrix M
       
    if inttype == "mutualistic":
        M = abs(M) # All values are positive (half-normal distribution)
        
    if inttype == "trophic":
        M = abs(np.tril(M)) - abs(np.triu(M)) 
        # Lower triangle elements are positive (effect of prey on predators)
        # Upper triangle elements are negative (effect of predators on prey)
    
    maskint = np.eye(A.shape[1], dtype=bool) # Diagonal elements
    M[:,maskint] = -d # Species self-regulation set to -d
   
    #### Observed eigenvalues ####
            
    eigv,eigvect = np.linalg.eig(M) # Eigenvalues and eigenvectors
    eig = np.amax(eigv.real, 1) # Maximal real part of the eigenvalues
    
    #### Predicted eigenvalues ####
    
    S = A.shape[0] # Number of species
    L = np.sum(np.tril(A, k=-1)) # Number of links
    b = np.log(L)/np.log(0.5*S) # Shape of the L~S relationship
    alpha = sum(np.sum(np.tril(A), 
                       axis=1)==0)/S # Proportion of the first partition
    pred = predEig(S, b, inttype, alpha, mu, sigma2, d) # Predicted R(eig)_max
    res = namedtuple('Resires', ('obs', 'var', 'pred'))
    return(res(eig.mean(), eig.var(), pred))

##############################################################################
# 'Private' function
##############################################################################

def predEig(S, b, inttype, alpha, mu=0, sigma2=1, d = 0, rho=(2/np.pi)):
    """
    Computes the predicted real part of the rightmost eigenvalue.
    
    Parameters
    ----------
    S : integer
        Number of species.
        
    b : float
        Shape of the links~species relationship given by b = log(L)/log(S/2).
        
    inttype : string
        Type of the interactions (either 'trophic' or 'mutualistic').
        
    alpha : float
        Fraction of species belonging to the first partition of bipartite networks.
        
    mu : float
        Mean interaction strength (0 by default).
    
    sigma2 : float
        Variance of interaction strength (1 by default).
    
    d : float or list
        Species self-regulation (diagonal elements of M = -d; 0 by default).

    rho : float
        Correlation coefficient of pairwise interactions.
    
    Returns
    -------
    Predicted real part of the rightmost eigenvalue based on Equation 7-8.
    
    Notes
    ----------
    Explanations are available in the Methods and in Supplementary Equations 4.
    """
    
    #### Connectance ####
    
    C = 2**(1-b)*S**(b-2) # Connectance of the community matrix
    
    #### Trophic networks ####
    
    if inttype=="trophic":
        eigP = -d + (1-rho)*((sigma2*S*C)**0.5) # Equation 7
        return(eigP)
    
    #### Mutualistic networks ####
    
    if inttype == "mutualistic":
        
        ### Half-normal distribution ###
        
        sigma2h = sigma2 *(1-(2/np.pi))# Variance
        muh = mu + sigma2 * (2/np.pi)**0.5 # Mean
        
        ### Between-subsytems metrics ###
        
        Cb = C/(2*alpha*(1-alpha))# Between-subsystem connectance 
       
        mub = Cb*muh # Between-subsystem mean 
        
        varb = Cb*(sigma2h+(1-Cb)*muh**2) # Between-subsystem variance
        
        rhob = (rho*sigma2h+(1-Cb)*muh**2
                )/(sigma2h+(1-Cb)*muh**2)# Between-subsystem correlation
        
        ### Bulk ###
        center = S*varb*rhob # Center
        xaxis = S*varb*((alpha*(1-alpha))**0.5)*(1+rhob**2) # Radius
        bulk = (center + xaxis)**0.5 # Bulk
        
        ### outlier ###
        lambdaD1 = S*mub*(alpha*(1-alpha))**0.5
        outlier = lambdaD1+((rho*S*varb)/lambdaD1)
        
        ### Eigenvalue ###
        eigP = -d +np.max([bulk, outlier],axis=0)
        
    return(eigP)  
