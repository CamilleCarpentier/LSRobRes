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
from scipy.stats import linregress
from collections import namedtuple

##############################################################################
# 'Public' functions
##############################################################################


# Decompositions (in-silico extinction experiments)
###################################################

def experiment(mat, nbsimu, independent = False):
    """
    Simulates the network decompositions (in-silico extinction experiments)

    Parameters
    ----------
    mat : numpy array of shape (S,S), S being the number of species
        Adjacency matrix of the network.
        
    nbsimu : integer
        Number of simulations (decompositions) to perform.
            
    independent : bool
        Should the species having no incoming links be considered as 
        independent (i.e. not undergo secondary extinction)?

    Returns
    -------
    S: integer
        Number of species in the network  
        
    L: integer
        Number of links (edges) in the network (excluding cannibalism)
        
    b: float
        Shape of the L~S relationship defined as log(L)/log(S/2)
        
    z: float
        Proportion of basal species
        
    sseq: numpy array of shape (nbsimu, S)
        Number of species along each decomposition 
        (one row is one whole decomposition of the network)
   
    lseq: numpy array of shape (nbsimu, S)
        Number of links along each decomposition 
        (one row is one whole decomposition of the network)

    """
    
    #-------- Network structure -------- 
    mat = np.tril(mat != 0) # Binary matrix of incoming links (low triangle)
    S = mat.shape[0] # Number of species
    L = np.sum(np.tril(mat, k = -1)) # Number of edges
    b = np.log(L)/np.log(0.5*S) # Shape of the L~S relationship

    #-------- Wanted matrix and Independent species -------- 
    mat = trophorder(mat) # Order the species by "trophic" level in the matrix 
    z = sum(np.sum(mat, axis=1)==0) # Number of independent species
    # Independent sp. are considered as sp. having no incoming links
    
    #-------- Initial values for each simulation -------- 
    # Number of species
    sseq = [nbsimu*[S]] # All decompositions start with all species
    # Number of links
    lseq = [nbsimu*[L]] # All decompositions start with all links
    # Adjacency matrix
    adj = np.array(nbsimu*[mat]) # All decompositions start with the same adj
    # Presence/Absence of species
    species = np.ones((nbsimu, S)) # At the start, all species are present
    
    #-------- Network decompositions -------- 
    
    while np.sum(adj)!= 0 : # Simulations run until there is no link left
        
        ### Random species removal ###
        species = removal(species, nbsimu) # One species removed in each simulation
        adj = cancel(adj, species) # Update of the adjacency matrices
       
        ### Secondary extinctions ###
        species = extinction(species, adj, z, independent)
        # Update of the presence/absence of each species in each simulation
        
        ### Save decomposition sequence ###
        sseq.append(np.sum(species, axis=1)) # Storage of the number of species
        lseq.append(np.sum(np.sum(
            np.tril(cancel(adj, species)), axis=2), axis=1))
            # Storage of the number of links
    
    sseq = np.array(sseq).T # Each row is a simulation, each column is a step
    lseq = np.array(lseq).T # Each row is a simulation, each column is a step
    result = namedtuple('experes', ('S', 'L', 'b', 'z', 'sseq', 'lseq'))
    if not independent :
        z = 0
    return(result(S, L, b, z/S, sseq, lseq))


# R2 of the L~S relationship prediction
########################################

def R2L(sseq, lseq, b):
    """
    Computes the coefficient of determination of the regression and the b-value based 
    approaches to predict the number of links along the network decomposition. 
    
    Parameters
    ----------
    sseq : numpy array of shape (nbsimu, S) with nbsimu being the number of network decompositions done.
        Each row of this array contains the species richness along one decomposition.
        
    lseq : numpy array of shape (nbsimu, S) with nbsimu being the number of network decompositions done.
        Each row of this array contains the number of links along one decomposition.
        
    b : float
        Shape of the links~species relationship given by b = log(L)/log(S/2).
    
    Returns
    -------
        
    breg : float
        Estimation of b (from L = a * S^b) based on log-log regression
        
    areg : float
        Estimation of a (from L = a * S^b) based on log-log regression
    
    r2reg : float
        R2 of the predictions of L based on the log(L)~log(S) regression
    
    r2b : float
        R2 of the predictions of L based on the log(L) = b*log(S/2) equation
    
    """
    
    #### Log-log regression-based approach ####
    lseqforl  = lseq[(sseq!=0) & (lseq!=0)] # Non-zero value only in log space
    sseqforl = sseq[(sseq!=0) & (lseq!=0)] # Non-zero value only in log space
    paramL = linregress(np.log(sseqforl).flat, np.log(lseqforl).flat) # Regression
    
    #### b-value based approach ####
    obs = np.log(lseqforl) # Observed values
    pred = b*np.log(sseqforl/2) # Predicted values
    SSres = np.sum((obs-pred)**2) # Residual sum of squares
    SStot = np.sum((obs-np.mean(obs))**2) # Total sum of squares
    r2Lb = 1 - (SSres/SStot) # Coefficient of determination
    
    result = namedtuple('R2Lres', ('breg','areg','r2reg', 'r2b'))
    return(result(paramL.slope, np.exp(paramL.intercept),
                  paramL.rvalue, r2Lb))


##############################################################################
# 'Private' functions  
##############################################################################

def trophorder(mat):
    """ 
    Order the matrix based on "trophic level": the species having no incoming 
    links are in the first rows/columns; the species having the highest number 
    of incoming links is in the last row.
    
    Parameters
    ----------
    mat : numpy array of shape (S, S)
        the binary matrix containing ones (if there is an incoming link) and zeros otherwise.
    
    Returns
    -------
    Numpy array of shape (S, S) containing the matrix 'mat' ordered.
    
    """
    #------- Getting each species trophic level -------
    S = mat.shape[0] # Number of species
    level = np.repeat(True, S)
    troph = np.zeros((S,)) # Each species starts at the level 0 (i.e. basal)
    for l in np.arange(S):
        troph += mat[:,level].sum(1) != 0 # Do species feed on the lower level?
        level = troph > l 
        # If a sp. feeds on lower level, it might belong to the next level
    #------- Ordering based on trophic level -------
    to_order = troph.argsort() # Order by the level
    newmat = np.full_like(mat, 0) # New matrix to fill in
    for sp1 in np.arange(mat.shape[0]): # For each row
            for sp2 in np.arange(mat.shape[0]): # For each columns
                newmat[sp1,sp2] = mat[to_order[sp1],to_order[sp2]]
    return(newmat)  

def removal(species, nbsimu):
    """
    Removes one extant species from the network for each simulation (decomposition).

    Parameters
    ----------
    species : numpy array of shape (nbsimu, S) with nbsimu being the number of simulations (decompositions).
        This array contains the information about the presence (1) or 
        absence (0) of each species (columns) in each simulation (rows).
        
    nbsimu : integer
        Number of simulations to perform.

    Returns
    -------
    Numpy array of shape (nbsimu, S) with one species removed per simulation.

    """
    for n in range(nbsimu): # For each simulation
        extant = np.where(species[n,:] != 0)[0] # Localise extant species
        if len(extant) != 0: # If there is still at leats one species left
            # Random species switch from present to absent
            toremove = np.random.permutation(extant)[0] # Draw one random sp.
            species[n, toremove] = 0 # Removal of the drawn species.
    return(species)


def cancel(adj, species):
    """
    Removes links that are in- or outgoing from species that are not 
    in the network anymore.

    Parameters
    ----------
    adj : numpy array of size (S,S) with S being the species richness
        Adjacency matrix.
        
    species : numpy array of shape (nbsimu, S) with nbsimu being the number of simulations (decompositions).
        This array contains the information about the presence (1) or 
        absence (0) of each species (columns) in each simulation (rows).
        
    Returns
    -------
    Numpy array of shape (S,S) corresponding to the adjacency matrix 
    without the links that needed to be removed.

    """
    # Row i full of 0 if the species i is extinct and full of 1 otherwise
    cancelrow = np.repeat(species, species.shape[1],axis=1).reshape(*adj.shape)
    # Column j full of 0 if the species j is extinct and full of 1 otherwise
    cancelcol = np.repeat(species, species.shape[1],axis=0).reshape(*adj.shape)
    adj = cancelcol * cancelrow * adj
    return(adj)

def extinction(species, adj, z, independent):
    """
    Returns the presence/absence of each species after taking into account 
    the secondary extinctions.
    
    Parameters
    ----------
    species : numpy array of shape (nbsimu, S) with nbsimu being the number
        of simulations (decompositions). This array contains the information 
        about the presence (1) or absence (0) of each species (columns) in 
        each simulation (rows).

    adj : numpy array of size (S,S) with S being the species richness
        Adjacency matrix.
        
    z : float
        Number of species which might not undergo secondary extinction.

    independent : bool
        Should the species having no incoming links be considered as 
        independent (i.e. not undergo secondary extinction)?

    Returns
    -------
    Numpy array of shape (nbsimu, S) containing, for each decomposition (row), 
    the presence (1) or absence (0) of each species (columns).

    """
    #-------- Extinction of dependent species --------
    # Basic rule for dependent species : 
    # they need to be linked to another species to be part of the network
    left = np.sum(adj, axis = 2)[:,z:] # Number of neighbours left
    Psurvival = (left > 0).astype(int) # Survival if at least 1 neighbour

    # Extinction cascade through trophic levels
    while np.sum(species[:,z:] != Psurvival) != 0 :
        
        ### Extinction(s) ###
        # Removal of non surviving species
        species[:,z:] = (species[:,z:])*Psurvival 
        # Removal of non surviving links (i.e. links of the extinct species)
        adj = cancel(adj, species)
        
        ### Check for higher order extinctions ###
        left = np.sum(adj, axis=2)[:,z:] # Number of neighbours left
        Psurvival = (left > 0).astype(int) # Survival if at least 1 neighbour

    #-------- Extinction of independent species --------
    if independent==False: # If there is no independent species
        # Species having no incoming link undergo secondary extinction
        interact = np.sum(cancel(adj, species),axis=1)[:, :z] # Outgoing links
        (species[:,:z])[interact == 0] = 0 # Removed if no outgoing links left
        
    return(species)
