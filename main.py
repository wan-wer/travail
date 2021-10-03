import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import *
import colorsys
from collections import defaultdict
from scipy.cluster.vq import kmeans2
def readfile(filename):
    with open(filename) as f:
        data = f.read().splitlines()
    return  data

class Population:
    def __init__(self):
        self.data = np.loadtxt('population_donnees.txt')
        self.nom_individus = readfile('population_noms_individus.txt')
        self.nom_variables = readfile('population_noms_variables.txt')

def acp(X):
    I = X.shape[0]
    K = X.shape[1]
    D = np.eye(I)/I
    M = np.eye(K)

    Xcov_ind = X.T.dot(D.dot(X.dot(M)))
    Val, Vect = np.linalg.eig(Xcov_ind)

    indices = np.argsort(Val)[::-1]
    val_p_ind = np.sort(Val)[::-1]
    vect_p_ind = Vect[:,indices]

    fac_ind =X.dot(M.dot(vect_p_ind))
    return fac_ind

def acm(data):
    nb_modalites_par_var = data.max(0)
    nb_modalites = int(nb_modalites_par_var.sum())

    XTDC = np.zeros((data.shape[0], nb_modalites))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            XTDC[i, int(nb_modalites_par_var[:j].sum() + data[i, j] - 1)] = 1
    Xfreq = XTDC / XTDC.sum()
    marge_colonne = Xfreq.sum(1).reshape(Xfreq.shape[0], 1)
    # print(marge_colonne.shape)
    marge_ligne = Xfreq.sum(0).reshape(1, Xfreq.shape[1])
    # print(marge_ligne.shape)

    Xindep = marge_ligne * marge_colonne
    Xindep[Xindep==0] = -1
    # print(Xindep.shape)


    X = Xfreq / Xindep - 1

    M = np.diag(marge_ligne[0, :])
    D = np.diag(marge_colonne[:, 0])
    # print(M.shape)
    # print(D.shape)

    # Calcul de la matrice de covariance pour les modalités en ligne
    Xcov_ind = X.T.dot(D.dot(X.dot(M)))

    # Calcul des valeurs et vecteurs propres de la matrice de covariance
    L, U = np.linalg.eig(Xcov_ind)

    # Tri par ordre décroissant des valeurs des valeurs propres
    indices = np.argsort(L)[::-1]
    val_p_ind = np.float16(np.sort(L)[
                           ::-1])  # car des valeurs peuvent être complexes avec une partie imaginaire nulle en raison d'approximations numériques
    vect_p_ind = U[:, indices]

    # Suppression des éventuelles valeurs propres nulles
    indices = np.nonzero(val_p_ind > 0)[0]
    val_p_ind = val_p_ind[indices]
    vect_p_ind = vect_p_ind[:, indices]

    # Calcul des facteurs pour les modalités en ligne
    fact_ind = X.dot(M.dot(vect_p_ind))

    # Calcul des facteurs pour les modalités en colonne
    fact_mod = X.T.dot(D.dot(fact_ind)) / np.sqrt(val_p_ind)


    return fact_ind


def CAH(fac_ind, me = 'ward', t = 8.1):
    Z = linkage(fac_ind, method=me)
    plt.figure()
    dendrogram(Z)

    return fcluster(Z, t, criterion='distance')

def centre_mobile(fac_ind, k = 3, me = 'points'):
    centroid, label = kmeans2(fac_ind, k, minit = me)
    return centroid, label



def dessin_CAH(fac_ind, fc, nom_individus):
    fc_max = max(fc)
    fac_ind = fac_ind[:,:2]
    cate = defaultdict(list)
    for count, value in enumerate(fc):
        cate[value].append(fac_ind[count])
    centre = defaultdict(list)
    for i in range(1, (fc_max + 1)):
        centre[i] = np.mean(cate[i],axis=0)
    colors = _get_colors(fc_max)
    plt.figure()
    for i in range(1, (fc_max + 1)):
        plt.scatter(centre[i][0],centre[i][1], marker = 'X', s = 200, color = colors[i-1])
        x = np.array(cate[i])[:,0]
        y = np.array(cate[i])[:,1]

        plt.scatter(x, y, marker = 'o', s = 20, color = colors[i-1])
    for label,x,y in zip(nom_individus, fac_ind[:,0], fac_ind[:,1]):
        plt.annotate(label, xy = (x,y), xytext = (-50, 5),textcoords = 'offset points', arrowprops = dict(arrowstyle='->',connectionstyle = 'arc3,rad = 0'))
    plt.show()

def dessin_centre_mobile(fac_ind, centroid, label, nom_individus):
    fc = np.array(label) + 1
    fc_max = max(fc)

    colors = _get_colors(fc_max)
    fac_ind = fac_ind[:, :2]
    cate = defaultdict(list)
    for count, value in enumerate(fc):
        cate[value].append(fac_ind[count])
    for i in range(1, (fc_max + 1)):
        plt.scatter(centroid[i-1,0], centroid[i-1, 1], marker='X', s=200, color=colors[i - 1])
        x = np.array(cate[i])[:, 0]
        y = np.array(cate[i])[:, 1]

        plt.scatter(x, y, marker='o', s=20, color=colors[i - 1])
    for label,x,y in zip(nom_individus, fac_ind[:,0], fac_ind[:,1]):
        plt.annotate(label, xy = (x,y), xytext = (-50, 5),textcoords = 'offset points', arrowprops = dict(arrowstyle='->',connectionstyle = 'arc3,rad = 0'))
    plt.show()




def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

def ACM_CAH(pp, t = 0.452, me = 'ward'):
    fac_ind = acm(pp.data)
    fac_ind = np.real(fac_ind)
    fc = CAH(fac_ind, t=t, me=me)
    dessin_CAH(fac_ind, fc, nom_individus=pp.nom_individus)

def ACM_CM(pp, k=3):
    fac_ind = acm(pp.data)
    fac_ind = np.real(fac_ind)
    centroid, label = centre_mobile(fac_ind, k=k)
    dessin_centre_mobile(fac_ind, centroid, label, nom_individus=pp.nom_individus)

def ACP_CAH(pp,t = 8, me = 'ward'):
    moy = pp.data.mean(0)
    var = pp.data.std(0)
    pp.data = (pp.data - moy) / var
    fac_ind = acp(pp.data)
    fc = CAH(fac_ind, t=t, me=me)
    dessin_CAH(fac_ind, fc, nom_individus=pp.nom_individus)

def ACP_CM(pp, k=3):
    moy = pp.data.mean(0)
    var = pp.data.std(0)
    pp.data = (pp.data - moy) / var
    fac_ind = acp(pp.data)
    centroid, label = centre_mobile(fac_ind, k=k)
    dessin_centre_mobile(fac_ind, centroid, label, nom_individus=pp.nom_individus)


if __name__ == '__main__':
    pp = Population()
    ACP_CM(pp)
















