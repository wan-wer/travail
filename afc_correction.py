#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 09:45:12 2021

@author: dellandrea
"""

import numpy as np
import matplotlib.pyplot as plt

def readfile(filename):
	with open(filename) as f:
		data = f.read().splitlines()

	return data

def afc(data,noms_modalites1,noms_modalites2):
    # print(data)
    # print(data.shape)
    # print(noms_modalites1)
    # print(noms_modalites2)
    
    Xfreq = data / data.sum()
    
    marge_colonne = Xfreq.sum(1).reshape(Xfreq.shape[0],1)
    # print(marge_colonne.shape)
    marge_ligne = Xfreq.sum(0).reshape(1,Xfreq.shape[1])
    # print(marge_ligne.shape)
    
    Xindep = marge_ligne * marge_colonne
    # print(Xindep.shape)
    
    X = Xfreq / Xindep - 1
    
    M = np.diag(marge_ligne[0,:])
    D = np.diag(marge_colonne[:,0])
    # print(M.shape)
    # print(D.shape)
    
    # Calcul de la matrice de covariance pour les modalités en ligne
    Xcov_ind = X.T.dot(D.dot(X.dot(M)))
    
    # Calcul des valeurs et vecteurs propres de la matrice de covariance
    L,U = np.linalg.eig(Xcov_ind)
    
    # Tri par ordre décroissant des valeurs des valeurs propres
    indices = np.argsort(L)[::-1]
    val_p_mod1 = np.sort(L)[::-1]
    vect_p_mod1 = U[:,indices]
    
    # Calcul des facteurs pour les modalités en ligne 
    fact_mod1 = X.dot(M.dot(vect_p_mod1))
    
    # Calcul des facteurs pour les modalités en colonne
    fact_mod2 = X.T.dot(D.dot(fact_mod1)) / np.sqrt(val_p_mod1)
    
    # Calcul des pourcentage d'inertie des axes factoriels
    inerties = 100*val_p_mod1 / val_p_mod1.sum()
    
    print('Pourcentages d"inertie :')
    print(inerties)
    
    # Affichage du diagramme d'inertie
    plt.figure(1)
    plt.plot(inerties,'o-')
    plt.title('Diagramme des inerties')
    
    
    # Calcul de la contribution des modalités en ligne
    contributions_mod1 = np.zeros(fact_mod1.shape)
    for i in range(fact_mod1.shape[1]):
        f = fact_mod1[:,i]
        contributions_mod1[:,i] = 100 * D.dot(f*f) / f.T.dot(D.dot(f))
        
    print('Contribution des modalités en ligne :')
    print(contributions_mod1)

    # Calcul de la contribution des modalités en colonne
    contributions_mod2 = np.zeros(fact_mod2.shape)
    for i in range(fact_mod2.shape[1]):
        f = fact_mod2[:,i]
        contributions_mod2[:,i] = 100 * M.dot(f*f) / f.T.dot(M.dot(f))
        
    print('Contribution des modalités en colonne :')
    print(contributions_mod2)    
    
    # Calcul de la qualité de représentation des modalités en ligne
    distance = (fact_mod1**2).sum(1)
    distance = distance.reshape(fact_mod1.shape[0],1)
    
    qualite_mod1 = fact_mod1**2 / distance
    
    print('Qualité de représentation des modalités en ligne :')
    print(qualite_mod1)
    
    # Calcul de la qualité de représentation des modalités en colonne
    distance = (fact_mod2**2).sum(1)
    distance = distance.reshape(fact_mod2.shape[0],1)
    
    qualite_mod2 = fact_mod2**2 / distance
    
    print('Qualité de représentation des modalités en colonne :')
    print(qualite_mod2)

    # Affichage du plan factoriel pour les modalités en ligne
    plt.figure(2)
    plt.plot(fact_mod1[:,0],fact_mod1[:,1],'x')
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('AFC Projection des modalités en ligne')
    for label,x,y in zip(noms_modalites1,fact_mod1[:,0],fact_mod1[:,1]):
        plt.annotate(label,
                     xy=(x,y),
                     xytext=(-50,5),
                     textcoords="offset points",
                     arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0')
                     )
    
    # Affichage du plan factoriel pour les modalités en colonne
    plt.figure(3)
    plt.plot(fact_mod2[:,0],fact_mod2[:,1],'x')
    plt.grid(True)
    plt.axvline(linewidth=0.5,color='k')
    plt.axhline(linewidth=0.5,color='k')
    plt.title('AFC Projection des modalités en colonne')
    for label,x,y in zip(noms_modalites2,fact_mod2[:,0],fact_mod2[:,1]):
        plt.annotate(label,
                     xy=(x,y),
                     xytext=(-50,5),
                     textcoords="offset points",
                     arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0')
                     )
    
    plt.show()    
    

if __name__ == '__main__' :
    
    # Lecture des données à partir des fichiers texte
    data = np.loadtxt('TD3-donnees/csp-donnees.txt')
    noms_modalites1 = readfile('TD3-donnees/csp-noms_modalites1.txt')
    noms_modalites2 = readfile('TD3-donnees/csp-noms_modalites2.txt')
        
    # Réalisation de l'afc
    afc(data,noms_modalites1,noms_modalites2)