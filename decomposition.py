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
from scipy.stats import linregress
from collections import namedtuple

##############################################################################
# 'Public' functions
##############################################################################


# Decompositions (in-silico extinction experiments)
###################################################

def experiment(mat, nbsimu, directed, basal, triangle = "L"):
    """
    Simulates the network decompositions (in-silico extinction experiments)

    Parameters
    ----------
    mat : numpy array of shape (S,S), S being the number of species
        Adjacency matrix of the network.
        
    nbsimu : integer
        Number of simulations (decompositions) to perform.
        
    directed : bool
        Either the decomposition should be performed on 
        the directed graph (True) or not (False).
        
    basal : bool
        Either basal species should undergo secondary extinction (True) 
        or not (False).
    
    triangle : string
        "L" if the links are oriented from j to i (i as row, j as column);
        "U" if the links are oriented from i to j (i as row, j as column).

    Returns
    -------
    S: integer
        Number of species in the network  
        
    L: integer
        Number of links (edges) in the network
        
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
    
    #### Network structure ####
    
    S = mat.shape[0]# Number of species
    L = np.sum(np.tril(mat!=0)) # Number of edges
    b = np.log(L)/np.log(0.5*S) # Shape of the L~S relationship

    #### Wanted matrix and basal species ####
   
    if directed:
        if triangle == "U":
            mat = np.triu(mat!=0)
        else:
            mat = np.tril(mat!=0)
        # Only lower triangle elements are used for the directed graph approach
        # (unless it is specified that the upper triangle should be used)
    
    mat = trophorder(mat) # Order the species by "trophic" level in the matrix 
    if directed :
        z = sum(np.sum(mat, axis=1)==0)# Number of basal species
    else :
        z = 0 # No basal species if undirected graph
    
    #### Initial values for each simulation ####
    
    # Number of species
    sseq = [nbsimu*[S]] # All decompositions start with all species
    # Number of links
    lseq = [nbsimu*[L]] # All decompositions start with all links
    # Adjacency matrix
    adj = np.array(nbsimu*[mat!=0]) # All decompositions start with the same adj
    # Presence/Absence of species
    species = np.ones((nbsimu,S)) # At the start, all species are present
    
    #### Network decompositions ####
    
    while np.sum(adj)!= 0 : # Simulations run untill there is no species left
        
        ### Random species removal ###
        species = removal(species, nbsimu) # One species removed in each simulation
        adj = cancel(adj, species) # Update of the adjacency matrices
       
        ### Secondary extinctions ###
        species = extinction(species, adj, z, basal)
        # Update of the presence/asbence of each species in each simulation
        
        ### Save decomposition sequence ###
        sseq.append(np.sum(species, axis=1)) # Storage of the number of species
        lseq.append(np.sum(np.sum(
            np.tril(cancel(adj, species)), axis=2), axis=1))
            # Storage of the number of links
    
    sseq = np.array(sseq).T # Each row is a simulation, each column is a step
    lseq = np.array(lseq).T # Each row is a simulation, each column is a step
    result = namedtuple('experes', ('S', 'L', 'b', 'z', 'sseq', 'lseq'))
    if basal:
        z = 0 # For the following computations, z = 0 if basal True 
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
        R2 of the predictions of L based on the L = b*log(b/2) equation
    
    """
    
    #### Log-log regression-based approach ####
    lseqforl  = lseq[(sseq!=0) & (lseq!=0)] # Non-zero value only in log space
    sseqforl = sseq[(sseq!=0) & (lseq!=0)] # Non-zero value only in log space
    paramL = linregress(np.log(sseqforl).flat, np.log(lseqforl).flat)
    
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
    # Getting each species trophic level
    S = mat.shape[0]
    level = np.repeat(True, S)
    troph = np.zeros((S,))
    for l in np.arange(S):
        troph+=mat[:,level].sum(1) != 0
        level = troph > l
    #Ordering based on trophic level  
    to_order = troph.argsort()
    newmat = np.full_like(mat, 0)
    for sp1 in np.arange(mat.shape[0]):
            for sp2 in np.arange(mat.shape[0]):
                newmat[sp1,sp2] = mat[to_order[sp1],to_order[sp2]]
    return(newmat)  

def removal(species, nbsimu):
    """
    Removes one extant species from the network for each simulation (decomposition).

    Parameters
    ----------
    species : numpy array of shape (nbsimu, S) with S being the species richness
        This array contains the information about the presence (1) 
        or absence (0) of each species (columns) in each simulation (rows).
        
    nbsimu : integer
        Number of simulations to perform.

    Returns
    -------
    Numpy array of shape (nbsimu, S) with one species removed per simulation.

    """
    for n in range(nbsimu): # For each simulation
        extant = np.where(species[n,:] != 0)[0] # Localise extant species
        if len(extant)!=0: #If there is still at leats one species left
            # Random species switch from present to absent
            toremove = np.random.permutation(extant)[0]
            species[n, toremove] = 0
    return(species)


def cancel(adj, species):
    """
    Removes links that are in- or outgoing from species that are not 
    in the network anymore.

    Parameters
    ----------
    adj : numpy array of size (S,S) with S being the species richness
        Adjacency matrix.
        
    species : numpy array of shape (nbsimu, S) with nbsimu being the number of simulations (decompositions)
        This array contains the information about the presence (1) 
        or absence (0) of each species (columns) in each simulation (rows).
        
    Returns
    -------
    Numpy array of shape (S,S) corresponding to the adjacency matrix 
    without the links that needed to be removed.

    """
    cancelcol = np.repeat(species, species.shape[1], 
                          axis=0).reshape(*adj.shape)
    cancelrow = np.repeat(species, species.shape[1], 
                          axis=1).reshape(*adj.shape)
    adj = cancelcol * cancelrow * adj
    return(adj)

def extinction(species, adj, z, basal):
    """
    Returns the presence/absence of each species after taking into account 
    the secondary extinctions.
    
    Parameters
    ----------
    species : numpy array of shape (nbsimu, S) with nbsimu being the number of simulations (decompositions)
        This array contains the information about the presence (1) 
        or absence (0) of each species (columns) in each simulation (rows).

    adj : numpy array of size (S,S) with S being the species richness
        Adjacency matrix.
        
    z : float
        Number of species which cannot undergo secondary extinction.

     basal : bool
        Either basal species should undergo secondary extinction (True) or not (False).

    Returns
    -------
    Numpy array of shape (nbsimu, S) containing, for each decomposition (row), 
    the presence (1) or absence (0) of each species (columns)

    """
    #### Extinction of non-basal species ####
      
    left = np.sum(adj, axis = 2)[:,z:] # Number of neighbours left
    Psurvival = (left > 0).astype(int) # Survival if at least one neighbour left

    while np.sum(species[:,z:]!= Psurvival) != 0 :
        (species[:,z:]) = (species[:,z:])*Psurvival # Removal of non surviving species
        adj = cancel(adj, species) # Removal of non surviving links
        left = np.sum(adj, axis=2)[:,z:] # Check for higher order extinctions
        Psurvival = (left > 0).astype(int)

    #### Exctinction of basal species ####
    if basal:
        interact = np.sum(cancel(adj, species),axis=1)[:, :z]
        # If a basal species does not have neighbour left, it goes extinct
        (species[:,:z])[interact == 0] = 0
        
    return(species)

