#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Julien Choukroun & Samy David & Jessica Gourdon & Luc Sagnes
"""


import numpy as np
import random
from numpy.linalg import *
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import math as mths
import copy as cp


borneInf = -1000
borneSup = 1000
Population = 2000
cas = 1000
dimension = 2
nbIteration = 1000

# Construit le nuage de points aléatoirement
def initialise(n, cas, borneInf, borneSup):
    x = np.zeros((n,cas))
    for i in range(n):
        for j in range(cas):
            x[i,j] = np.random.randint(borneInf, borneSup)
    return x

# Construction de la fonction test
def Rosenbrock(E):
    f = []
    n = np.shape(E)[0]
    for i in range(n-1):
        fi = 100*(E[1]-E[0]**2)**2+(1-E[0])**2
        f.append(fi)
    f=np.asarray(f)
    return f

# Calcule la différence des barycentres entre 2 points
def barycentre(xAncien, xNouveau, eps):
    baryAncien = np.zeros((1,np.shape(xAncien)[0]))
    baryNouveau = np.zeros((1,np.shape(xAncien)[0]))
    # On parcourt toutes les colonnes
    for i in range(np.shape(xAncien)[1]):
        baryAncien = baryAncien+xAncien[:,i]
        baryNouveau = baryNouveau+xNouveau[:,i]
    baryAncien = 1.0*baryAncien/np.shape(xAncien)[1]
    baryNouveau = 1.0*baryNouveau/np.shape(xNouveau)[1]
    norme = mths.sqrt(np.sum((baryNouveau-baryAncien)**2))
    return norme>eps

def PSO(E, f, nbIteration, w, phi1, phi2):
    # Initialisation
    #nb : Nombre de x0 pris 
    #V0 : vitesse initale de chaque particule (égale à 0) au début.
    #xAncien : nos xi tout au long de notre fonction
    #xLB : les local best de chaque particule
    #fLB : la fonction de Rosenbrock évaluée aux minimums locaux
    #fGB : la fonction de Rosenbrock évaluée à notre global best
    #xGB : notre minimum global
    nb = np.shape(E)[1]
    V0 = np.zeros(np.shape(E))
    xAncien = cp.deepcopy(E)
    xLB = cp.deepcopy(E)
    # On calcule l'image de xLB dans notre fonction
    fLB = f(xLB)
    # On cherche le minimum global xGB qui correspond au meilleur xLB
    fGB = np.min(fLB)
    index = np.where(fLB == fGB)
    index = index[1]
    xGB = xLB[:,index]
    # On fait les deux premières itérations de la boucle séparé pour pouvoir
    # nous servir de la différence de barycentre ensuite
    VAncien = V0
    # On calcule nos vitesses V_(i+1) puis nos positions x_(i+1)
    VSuiv = w*VAncien+phi2*random.random()*(xGB-x0)
    xSuiv = xAncien+VSuiv
    # On regarde ensuite si nos x_(i+1) sont mieux que nos xLB
    # Si tel est le cas, on change nos xLB des x_i en question
    for i in range(nb):
        fxi = f(xSuiv[:,i])
        if ((fxi < fLB).any()):
            xLB[:,i] = xSuiv[:,i]
    # On calcule l'image de xLB dans notre fonction
    fLB = f(xLB)
    # Si on a trouvé un nouveau global best, on le modifie avec l'ancien 
    fGBNouveau = np.min(fLB)
    if (fGBNouveau < fGB):
        fGB = fGBNouveau
        index = np.where(fLB == fGB)
        index = index[1]
        xGB = xLB[:,index]
    # On continue ensuite jusqu'à atteindre le maximum d'itération ou de trouver
    # une distance entre 2 barycentres assez petite entre nos x_(i+1) et nos x_i 
    k = 1
    while ((k <= nbIteration) and barycentre(xAncien, xSuiv, 10E-6)) :
        VAncien = cp.deepcopy(VSuiv)
        xAncien = cp.deepcopy(xSuiv)
        # Dans notre cas on doit faire varier phi2
        VSuiv = w*VAncien+phi1*random.random()*(xLB-xAncien)+phi2*random.random()*(xGB-xAncien)
        xSuiv = xAncien+VSuiv
        for i in range(nb):
            fxi = f(xSuiv[:,i])
            # On regarde si nos x_(i+1) sont mieux que nos xLB
            # Si tel est le cas, on change nos xLB des x_i en question
            if ((fxi < fLB).any()):
                xLB[:,i] = xSuiv[:,i]
        # On calcule l'image de xLB dans notre fonction
        fLB = f(xLB)
        # Si on a trouvé un nouveau global best, on le modifie avec l'ancien 
        fGBNouveau=np.min(fLB)
        if (fGBNouveau < fGB):
            fGB = fGBNouveau
            index = np.where(fLB == fGB)
            index = index[1]
            xGB = xLB[:,index]
        k=k+1
    plt.scatter(xLB[0,:],xLB[1,:],1,c='red')
    plt.ylim(-3000,3000)
    #plt.title('Essaim final')
    plt.title('Essaim initial (bleu) et essaim final (rouge)')
    print("Nombres itérations : ",k)
    # Meilleure solution trouvée xGB
    xMini = xGB
    fMini=f(xMini)
    return xMini,fMini


x0 = initialise(dimension, cas, borneInf, borneSup)
plt.scatter(x0[0,:],x0[1,:],1,c='blue')
#plt.title('Essaim initial')
v0 = np.zeros((dimension,cas))

f = Rosenbrock(x0)

xMini,fMini = PSO(x0, Rosenbrock, nbIteration, 1, 1, 1)
print("valeur de x min :",xMini)
print("Valeur de f min : ",fMini)
