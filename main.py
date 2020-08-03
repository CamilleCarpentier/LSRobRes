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
import matplotlib.pyplot as plt


##############################################################################
# Data and parameters
##############################################################################

#### Adjacency matrix of the network (example) ####

mat = np.array([[0,0,0,1,0,0],
                [0,0,0,0,1,1],
                [0,0,0,0,1,1],
                [1,0,0,0,1,1],
                [0,1,1,1,0,0],
                [0,1,1,1,0,0]])

# The lower triangle of mat should contain the incoming links:
# aij will be 1 when a link comes from j and goes to i
# (i being the row and j being the column).
# example: the 4th species eats the 1st one (the link is coming from 1st to 4th)
# while the 5th species eats the 2nd, the 3rd, the 4th species.


# The run time of this example goes up to 1 minute.
# The run time increases exponentially with the number of species
# (eg. 35 sp = 1 minute, 100 sp = 10 minutes, 110 sp = 25 minutes).


#### Parameters ####

inttype = "trophic" # Or "mutualistic".

directed = True # Should the decomposition be based on the directed graph:
# True: directed (lower triangle only, containing the incoming links); 
# False: undirected (full matrix).

basal = False # Do basal species undergo secondary extinctions:
# True: basal species go extinct when no neighbour left; 
# False: they remain in the network (unless randomly removed).

nbsimu = 10000 # Number of different decompositions to perform.



##############################################################################
# A new, network-specific approach
##############################################################################
# Note that the Figure 2 presented in the manuscript is based on the directed 
# graph with basal secondary extinctions (basal = True)
# while the example given here is without secondary extinction (basal = False).

import decomposition


# Decompositions (in-silico extinction experiments)
###################################################

S, L, b, z, sseq, lseq = decomposition.experiment(mat, nbsimu, directed, basal)

# S = Number of species.
# L = Number of links.
# b = Shape of the L~S relationship.
# z = Proportion of basal species.
# sseq = Number of species along each decomposition (row).
# lseq = Number of links along each decomposition (row).


# R2 of the prediction of the L~S relationship 
###############################################

breg, areg, r2reg, r2b = decomposition.R2L(sseq, lseq, b)

# breg = Estimation of b (from L = a * S^b) based on log-log regression.
# areg = Estimation of a (from L = a * S^b) based on log-log regression.
# r2reg = R2 of the prediction of lseq along sseq based on regression.
# r2b = R2 of the prediction of lseq along sseq based on the Equation 1.

print(r2reg, r2b)
# Computing r2reg et r2b for multiple networks allows to obtain
#                                                    Extended Data Figure 1.

# Figure 2 (equivalent)
#######################

#### Unique combinaison in the L~S plane ####
dots, likely = np.unique(tuple([sseq.flatten(), lseq.flatten()]), 
                          axis=1, return_counts=True)
likely[0] = nbsimu

#### Dots with size proportional to their likelihood in the L~S plane ####
plt.scatter(*dots, s = likely/100,  c="grey",
            label = "Observed")

#### Predictions of the L~S relationship ####
plt.plot(np.linspace(0,S,S*10), areg*np.linspace(0,S,S*10)**breg,
         c = "black", ls="dotted",
         label = "Log-log regression")# Regression
plt.plot(np.linspace(0,S,S*10),(np.linspace(0,S,S*10)/2)**b,
         c = "black",
         label = r"$L = (S/2)^b$") # Equation 1
plt.xlabel("Number of species (S)")
plt.ylabel("Number of links (L)")
plt.legend()


##############################################################################
# Predicting robustness
##############################################################################


import robustness


# Average number of extinctions occurring after the removal of one species
###########################################################################

deltaS, r2lost, lost, removed = robustness.R2deltaS(sseq, S, b, z)

# deltaS = Predicted average number of extinctions  after one removal (Equation 6).
# r2lost = R2 of the predictions of the number of species lost 
#               based on the number of species removed using deltaS.    
# lost = Number of species lost along each decomposition (row).
# removed = Number of species removed along each decomposition (row).
    
print(r2lost) 
# Computing r2lost for multiple networks allows to obtain 
#                                   Extended Data Figure 2b,c,d.


# Extended Data Figure 2a
##########################

#### Unique combinaison in the lost~removed plane ####
dots, likely = np.unique(tuple([removed.flatten(), lost.flatten()]), 
                          axis=1, return_counts=True)

#### Dots with size proportional to their likelihood in the lost~removed plane ####
plt.scatter(*dots, s = likely/100,  c="grey", label = "Observed")

#### Predictions of the lost~removed relationship ####
plt.plot(np.linspace(0,S,S*10), deltaS*np.linspace(0,S,S*10),
         c = "black", label = "Predicted")
plt.xlabel("Number of species removed (r)")
plt.ylabel("Number of species lost (n)")
plt.legend()


# Robustness
############

rob_obs, rob_var, rob_pred = robustness.robx(sseq, S, b, z, x=0.5)

# rob_obs = Mean robustness over nbsimu network decompositions.   
# rob_var = Observed variance of robustness over nbsimu network decompositions.
# rob_pred = Predicted robustness (Equation 7).

print(rob_obs, rob_pred)
# Computing the rob_obs for multiple networks allows to obtain Figure 3.


# Extended Data Figure 3a
##########################
xs = np.round(np.linspace(0.01,1,S),2) # Various robustness threshold
robs = []
for x in xs:
    robs.append([*robustness.robx(sseq, S, b, z, x)])
robs = np.array(robs)  
plt.errorbar(xs, robs[:,0], 
             yerr = robs[:,1]**0.5/2,
             c = "black", fmt='o', label = "Observed")  
plt.plot(xs, robs[:,2], c = "black", label = "Predicted", zorder=-1)
plt.xlabel("x")
plt.ylabel("Robustness at threshold x")
plt.legend()
# Computing rob_obs for multiple networks allows to obtain 
#                                                   Extended Data Figure 3b.


##############################################################################
# A Resilience - Robustness trade-off
##############################################################################

import resilience

res_obs, res_var, res_pred = resilience.realpart(mat, inttype, nbsimu = 1000)

# res_obs = Mean of the observed real part of the  rightmost eigenvalues.
# res_var = Variance of the observed real part of the rightmost eigenvalues.
# res_pred = Predicted real part of the  rightmost eigenvalue (Equation 8-9).

print(res_obs, res_pred)
# Computing res_obs for multiple networks allows to obtain:
            # The R2 of Resilience~b;
            # Figure 4.
