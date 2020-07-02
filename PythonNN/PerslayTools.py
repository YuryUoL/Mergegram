import sys
import numpy as np
import tensorflow as tf
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import sklearn_tda as tda
from tensorflow import random_uniform_initializer as rui
from perslay.experiments import single_run
import math

def FixDiagrams(diags):
    bounds = ComputeBounds(diags)
    maskedDiag = MaskDiagrams(diags)
    return NormalizeDiagram(maskedDiag, bounds)

def ComputeBounds(diags):
    dim = diags[0].shape[1]
    finalResults = np.zeros([2,dim])
    finalResults[0].fill(-float('inf'))
    finalResults[1].fill(float('inf'))
    for idx, it in enumerate(diags):
        for t in range(it.shape[0]):
            for u in range(it.shape[1]):
                finalResults[0][u] = max(finalResults[0][u], it[t][u])
                finalResults[1][u] = min(finalResults[0][u], it[t][u])

    return finalResults

# Should be in masked form:
def NormalizeDiagram(diags, bounds):
    it = diags
    for idx in range(it.shape[0]):
        for t in range(it.shape[1]):
            if it[idx][t][it.shape[2]-1] == 1:
                for u in range(it.shape[2]-1):
                    it[idx][t][u] = it[idx][t][u]/(bounds[0][u] -bounds[1][u]) - bounds[1][u]/(bounds[0][u] - bounds[1][u])
    return it


def MaskDiagrams(diags):
    dim = diags[0].shape[1]
    maxSize = 0;
    for it in diags:
        maxSize = max(maxSize, it.shape[0])
    ff = np.zeros([len(diags),maxSize,dim+1])
    for idx, it in enumerate(diags):
        poss = 0
        for rr in range(it.shape[0]):
            for coord in range(it.shape[1]):
                ff[idx][rr][coord] = it[rr][coord]
            ff[idx][rr][dim]=1
            poss = poss + 1
        while(poss < maxSize):
            for coord in range(it.shape[1]):
                ff[idx][poss][coord] = 0
            ff[idx][poss][dim] = 0
            poss = poss + 1
    return ff

    #maximums = np.zeros([dim])
    #minimums = np.zeros([dim])


def SingleDiagramCreator(input, thresh=500):
    diags_tmp = {"Diagram": input}
    # Whole pipeline
    tmp = Pipeline([
        ("Selector", tda.DiagramSelector(use=True, point_type="finite")),
        ("ProminentPts", tda.ProminentPoints(use=True, num_pts=thresh)),
        ("Scaler", tda.DiagramScaler(use=True, scalers=[([0, 1], MinMaxScaler())])),
        ("Padding", tda.Padding(use=True)),
    ])
    prm = {filt: {"ProminentPts__num_pts": min(thresh, max([len(dgm) for dgm in diags_tmp[filt]]))}
           for filt in diags_tmp.keys() if max([len(dgm) for dgm in diags_tmp[filt]]) > 0}

    # Apply the previous pipeline on the different filtrations.
    D = []
    for dt in prm.keys():
        param = prm[dt]
        tmp.set_params(**param)
        #print(diags_tmp[dt])
        D.append(tmp.fit_transform(diags_tmp[dt]))

    # For each filtration, concatenate all diagrams in a single array.
    diags = []
    for dt in range(len(prm.keys())):
        diags.append(np.concatenate([D[dt][i][np.newaxis, :] for i in range(len(D[dt]))], axis=0))

    return diags[0]


# def DiagramNormalizer(input, thresh):
#     tmp = Pipeline([
#         ("Selector", tda.DiagramSelector(use=True, point_type="finite")),
#         ("ProminentPts", tda.ProminentPoints(use=True, num_pts=thresh)),
#         ("Scaler", tda.DiagramScaler(use=True, scalers=[([0, 1], MinMaxScaler())])),
#         ("Padding", tda.Padding(use=True)),
#     ])
#     param = {"ProminentPts__num_pts": thresh}
#     tmp.set_params(**param)
#     return tmp.fit_transform(input)
#
def LoadData(path, name, numdiags, numpoints):
    list = []
    for x in range(0, numdiags):
        k = path + name + str(x) + ".npy"
        gaga = np.load(k)
        list.append(gaga)
    return SingleDiagramCreator(list, numpoints)

def LoadListData(paths, name, numdiags, numpoints):
    list = []
    for p in paths:
        for x in range(0, numdiags):
            k = p + name + str(x) + ".npy"
            gaga = np.load(k)
            list.append(gaga)
    return SingleDiagramCreator(list, numpoints)

def MultiDiagLoader(paths, subpaths, name, numdiags, numpoints):
    finalList = []
    for c in subpaths:
        newpaths = []
        for a in paths:
            newpaths.append(a + c + '/')
        finalList.append(LoadListData(newpaths,name,numdiags, numpoints))
    return finalList


def labelsGenerator(lbs, parts):

    labels = np.zeros(lbs)
    for i in range(0,lbs):
        a = math.floor(i / parts)
        labels[i] = a
    return labels

def labelsToVectorFormat(lbs, num_labels):
    output = np.zeros([lbs.shape[0],num_labels])
    for i in range(lbs.shape[0]):
        output[i][int(lbs[i])] = 1

    return output




def LoadListDataNew(paths, name, numdiags, dim):
    list = []
    for p in paths:
        for x in range(0, numdiags):
            k = p + name + str(x) + ".npy"
            gaga = np.load(k)
            list.append(gaga)
    return FixDiagrams(list)

def MultiDiagLoaderNew(paths, subpaths, name, numdiags, dim):
    finalList = []
    for c in subpaths:
        newpaths = []
        for a in paths:
            newpaths.append(a + c + '/')
        finalList.append(LoadListDataNew(newpaths,name, numdiags,dim))
    return finalList