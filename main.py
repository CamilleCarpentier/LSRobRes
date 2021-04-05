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
# example: here, the 4th species eats the 1st species 
# (the link is coming from 1st to 4th)
# while the 5th species eats the 2nd, the 3rd, the 4th species.


# The run time of this example goes up to 1 minute.
# The run time increases exponentially with the number of species
# (eg. 35 sp = 1 minute, 100 sp = 10 minutes, 110 sp = 25 minutes).


#### Parameters ####

inttype = "trophic" # Or "mutualistic".

independent = True # Should the species having no incoming links be condidered 
# as independant (i.e. not undergo secondary extinction)?
# False: All species might undergo secondary extinctions; 
# True: Species having no incoming links do not undergo secondary extinciton.

nbsimu = 10000 # Number of different decompositions (simulations) to perform.

##############################################################################
# Network specific L~S relationship
##############################################################################
# This part of the code relates to:
# - Equations 1-2
# - Figure 1
# - Extended Data Figure 1-2

# Note that the Figure 1 presented in the manuscript consider that all species
# undergo secondary extension (scenario 1 : independent = False) while the 
# example given here is without secondary extinction for the basal species
# (independant = True).

import decomposition


# Decompositions (in-silico extinction experiments)
###################################################

S, L, b, z, sseq, lseq = decomposition.experiment(mat, nbsimu, independent)

# S = Number of species.
# L = Number of links.
# b = Shape of the L~S relationship (based on Equation 2).
# z = Proportion of independent species.
# sseq = Number of species along each decomposition (row).
# lseq = Number of links along each decomposition (row).


# R2 of the prediction of the L~S relationship 
###############################################

breg, areg, r2reg, r2b = decomposition.R2L(sseq, lseq, b)

# breg = Estimation of b (from L = a * S^b) based on log-log regression.
# areg = Estimation of a (from L = a * S^b) based on log-log regression.
# r2reg = R2 of the prediction of lseq along sseq (in log-space) based 
#         on regression.
# r2b = R2 of the prediction of lseq along sseq (in log-space) based 
#       on the Equation 1.

print(r2reg, r2b)
# Computing r2reg et r2b for multiple networks allows to obtain
# Extended Data Figure 1.

# Figure 1 (equivalent)
#######################

#### Unique combinaison in the L~S plane ####
dots, likely = np.unique(tuple([sseq.flatten(), 
                                lseq.flatten()]), 
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
# This part of the code relates to:
# - Equations 3-5
# - Figure 2-3
# - Extended Data Figure 3a

import robustness


# Average number of extinctions occurring after the removal of one species
###########################################################################

deltaS, r2lost, lost, removed = robustness.R2deltaS(sseq, S, b, z)

# deltaS = Predicted average number of extinctions after one removal (Eq. 4).
# r2lost = R2 of the predictions of the number of species lost 
#          based on the number of species removed using deltaS.    
# lost = Number of species lost along each decomposition (row).
# removed = Number of species removed along each decomposition (row).
    
print(r2lost) 
# Computing r2lost for multiple networks allows to obtain Figure 2c.


# Figure 2a (equivalent)
##########################

#### Unique combinaison in the lost~removed plane ####
dots, likely = np.unique(tuple([removed.flatten(), 
                                lost.flatten()]), 
                          axis=1, return_counts=True)

#### Dots with size proportional to their likelihood in the lost~removed plane ####
plt.scatter(*dots, s = likely/100,  c="grey", label = "Observed")

#### Predictions of the lost~removed relationship ####
plt.plot(np.linspace(0,S,S*10), deltaS*np.linspace(0,S,S*10),
         c = "black", label = "Predicted")
plt.xlabel("Number of species removed (r)")
plt.ylabel("Number of species lost (n)")
plt.axhline(0.5*S, color = "black",
            linestyle = "dotted") # 50% of species lost
plt.legend()


# Robustness
############

rob_obs, rob_var, rob_pred = robustness.robx(sseq, S, b, z, x=0.5)

# rob_obs = Mean robustness over nbsimu network decompositions.   
# rob_var = Observed variance of robustness over nbsimu network decompositions.
# rob_pred = Predicted robustness (Equation 5).

print(rob_obs, rob_pred)
# Computing the rob_obs for multiple networks allows to obtain Figure 3.


# Extended Data Figure 3a
##########################
xs = np.round(np.linspace(0.01,1,S),2) # Various robustness threshold
robs = []
for x in xs: # For each threshold
    robs.append([*robustness.robx(sseq, S, b, z, x)]) # Compute robustness
robs = np.array(robs)  
plt.errorbar(xs, robs[:,0], 
             yerr = robs[:,1]**0.5/2,
             c = "black", fmt='o', label = "Observed")  
plt.plot(xs, robs[:,2], c = "black", label = "Predicted", zorder=-1)
plt.xlabel("x")
plt.ylabel("Robustness at threshold x")
plt.legend()
# Computing rob_obs for multiple networks allows to obtain 
# Extended Data Figure 3b.


##############################################################################
# Resilience - Robustness trade-off
##############################################################################
# This part of the code relates to:
# - Equations 7-8
# - Figure 5

import resilience

res_obs, res_var, res_pred = resilience.realpart(mat, inttype, nbsimu = 1000)

# res_obs = Mean of the observed real part of the  rightmost eigenvalues.
# res_var = Variance of the observed real part of the rightmost eigenvalues.
# res_pred = Predicted real part of the  rightmost eigenvalue (Equation 7-8).

print(res_obs, res_pred)
# Computing res_obs and res_pred for multiple networks allows to obtain:
# - The R2 of Resilience~b;
# - Figure 5.
