import pandas as pd
import numpy as np
import math

def entropy(Y: pd.Series,wt) -> float:
    """
    Function to calculate the entropy
    """
    
    total=np.sum(wt)
    map = dict() 
    for i in range(Y.size):
        if(Y.iat[i] in map):
            map[Y.iat[i]] =map[Y.iat[i]]+wt.iat[i]
            
        else:
            map[Y.iat[i]] = wt.iat[i]
     
    
    sum = 0
    p=0
    for i in map.keys():
        p = map[i]/(total)
        sum = sum-(p*np.log2(p))
    
    return sum



def gini_index(Y: pd.Series,wt) -> float:
    """
    Function to calculate the gini index
    """
    map = dict()
    total=np.sum(wt)
    for i in range(Y.size):
        if(Y.iat[i] in map):
            map[Y.iat[i]] = map[Y.iat[i]]+wt.iat[i]
        else:
            map[Y.iat[i]]= wt.iat[i]
    
    sum = 0
    p = 0
    for i in map.keys():
        p = map[i]/total
        sum = sum-p*p
    
    return sum



def information_gain(Y: pd.Series, attr: pd.Series,wt) -> float:
    """
    Function to calculate the information gain
    """
    assert(attr.size==Y.size)
    total=np.sum(wt)
    map={}
    info_gain=0
    for i in range(attr.size):
        if attr.iat[i] not in map:
            map[attr.iat[i]]=[[Y.iat[i],wt.iat[i]]]
        else:
            map[attr.iat[i]].append([Y.iat[i],wt.iat[i]])
    for i in map:
        map[i]=np.transpose(map[i])
        info_gain-=(np.sum(map[i][1])/total)*entropy(pd.Series(map[i][0]),pd.Series(map[i][1]))
    return info_gain

