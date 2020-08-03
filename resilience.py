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
# Main function
##############################################################################

def realpart(mat, inttype, mu = 0, sigma2 = 1, d = 1, nbsimu = 1000):
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
        If 'trophic', the aij and aji elments will be set by the function to 1 and -1.
    
    mu : float
        Mean interaction strength (0 by default).
    
    sigma2 : float
        Variance of interaction strength (1 by default).
    
    d : float or list
        Species self-regulation (diagonal elements of M = -d).
        If d is a list: the diagonal elements are drawn from a uniform distribution bounded by d[0] and d[1].
    
    Returns
    -------
    obs : float
        Mean of the observed real part of the  rightmost eigenvalue over nbsimu matrices.
        
    var : float
        Variance of the observed real part of the rightmost eigenvalue over nbsimu matrices. 
        
    pred : float
        Predicted real part of the  rightmost eigenvalue based on Equation 8-9.
        
    """

    #### Random matrix ####
    
    A = (mat!=0)# Adjacency matrix A
    W = np.random.normal(mu,sigma2,
                         size=(nbsimu, *A.shape)) # Interaction strength matrix
    M = W*A[np.newaxis,:,:] # Community matrix M
       
    if (inttype == "mutualistic") :
        M = abs(M) # All values are positive (half-normal distribution)    
    if (inttype == "trophic") :
        M = abs(np.tril(M)) - abs(np.triu(M)) 
        # Lower triangle elements are positive (effect of prey on predators)
        # Upper triangle elements are negative (effect of predators on prey)
    
    
    maskint = np.eye(mat.shape[1], dtype=bool) # Diagonal elements
    if type(d) == int:
        M[:,maskint] = -d # Species self-regulation set to -d
    else: 
        for n in range(nbsimu): # For each simulation
            M[n,maskint] = -np.random.uniform(d[0],d[1], 
                                                   size = mat.shape[0]) 
            # Species self-regulation drawn between -d[0] et -d[1] 
           
            
    #### Observed eigenvalues ####
            
    eigv,eigvect = np.linalg.eig(M) # Eigenvalues and eigenvectors
    eig = np.amax(eigv.real, 1) # Maximal real part of the eigenvalues
    
    #### Predicted eigenvalues ####
    
    S = A.shape[0] # Number of species
    L = np.sum(np.tril(A)) # Number of links
    b = np.log(L)/np.log(0.5*S) # Shape of the L~S relationship
    alpha = sum(np.sum(np.tril(A), 
                       axis=1)==0)/S # Proportion of the first partition
    pred = predEig(S, b, inttype, alpha, mu, sigma2, d) # Predicted R(eig)_max
    res = namedtuple('Resires', ('obs', 'var', 'pred'))
    return(res(eig.mean(), eig.var(), pred))

##############################################################################
# 'Private' function
##############################################################################

def predEig(S, b, inttype, alpha, mu=0, sigma2=1, d=1, rho=(2/np.pi)):
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
        Species self-regulation (diagonal elements of M = -d).
        If d is a list: the diagonal elements are drawn from a uniform distribution bounded by d[0] and d[1].

    rho : float
        Correlation coefficient of pairwise interactions.
    
    Returns
    -------
    Predicted real part of the rightmost eigenvalue based on Equation 8-9.
    
    Notes
    ----------
    The explanations of the formulas are available in Supporting Equations 2.
    """
    
    C = 2**(1-b)*S**(b-2) # Connectance of the community matrix
    
    #### Trophic networks ####
    
    if inttype=="trophic":
        eig = -d + (1-rho)*((sigma2*S*C)**0.5) # Equation 8
        return(eig)
    
    #### Mutualistic networks ####
    
    if inttype == "mutualistic":
        
        ### Half-normal distribution ###
        
        sigma2h = sigma2 *(1-(2/np.pi))# Variance
        muh = mu + sigma2 * (2/np.pi)**0.5 # Mean
        
        ### Between-subsytems metrics ###
        
        Cb = C/(2*alpha*(1-alpha))# Connectance
        varb = Cb*(sigma2h+(1-Cb)*muh**2)
        rhob = (rho*sigma2h+(1-Cb)*muh**2
                )/(sigma2h+(1-Cb)*muh**2)# Correlation of pairwise interactions
        
        # Eigenvalues ellipse
        center = S*varb*rhob
        xaxis = S*varb*(alpha*(1-alpha))**0.5*(1+rhob**2)
        bulk = -d+(center + xaxis)**0.5
        outlier = (S*C*muh)/(4*alpha*(1-alpha))**0.5
        eig = np.max([bulk, outlier],axis=0) 
     
        return(eig)