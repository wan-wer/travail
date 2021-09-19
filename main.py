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

def acp(X, nom_individus, nom_variables):
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

def CAH(fac_ind, me = 'ward', t = 400):
    Z = linkage(fac_ind, method=me)

    return fcluster(Z, t, criterion='distance')

def centre_mobile(fac_ind, k = 3, me = 'points'):
    centroid, label = kmeans2(fac_ind, k, minit = me)
    return centroid, label



def dessin_CAH(fac_ind, fc):
    fc_max = max(fc)
    fac_ind = fac_ind[:,:2]
    cate = defaultdict(list)
    for count, value in enumerate(fc):
        cate[value].append(fac_ind[count])
    centre = defaultdict(list)
    for i in range(1, (fc_max + 1)):
        centre[i] = np.mean(cate[i],axis=0)
    colors = _get_colors(fc_max)

    for i in range(1, (fc_max + 1)):
        plt.scatter(centre[i][0],centre[i][1], marker = 'X', s = 200, color = colors[i-1])
        x = np.array(cate[i])[:,0]
        y = np.array(cate[i])[:,1]

        plt.scatter(x, y, marker = 'o', s = 20, color = colors[i-1])
    plt.show()

def dessin_centre_mobile(fac_ind, centroid, label):
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
    plt.show()



    '''
    x = np.arange(-1, 1, 0.01)
    cercle_unite = np.zeros((2, len(x)))
    cercle_unite[0, :] = np.sqrt(1 - x ** 2)
    cercle_unite[1, :] = -cercle_unite[0, :]
    plt.plot(x, cercle_unite[0])
    plt.plot(x, cercle_unite[1])
    plt.plot(fac_var[:, 0], fac_var[:, 1], 'x')
    plt.yscale('linear')
    plt.grid(True)

    for label, x, y in zip(nom_variables, fac_var[:, 0], fac_var[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(-50, 5), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad = 0'))
    plt.show()
    '''

'''
    inertie = fac_ind.T.dot(D.dot(fac_ind))
    total = np.trace(inertie)
    pourcentage = list(range(inertie.shape[0]))
    for i in range(len(pourcentage)):
        pourcentage[i] = inertie[i,i]/total



    ##laoshi
    contribution_ind = np.zeros(fac_ind.shape)
    for i in range(fac_ind.shape[1]):
        f = fac_ind[:,i]
        contribution_ind[:, i] = 100 * D.dot(f*f)/f.T.dot(D.dot(f))


    distance = (fac_ind**2).sum(1)
    distance =distance.reshape(fac_ind.shape[0],1)
    qualite_ind = fac_ind**2/distance

'''

def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

if __name__ == '__main__':
    pp = Population()
    fac_ind = acp(pp.data, pp.nom_individus, pp.nom_variables)
    fc = CAH(fac_ind)
    #dessin_CAH(fac_ind, fc)
    centroid, label = centre_mobile(fac_ind)
    dessin_centre_mobile(fac_ind, centroid, label)





