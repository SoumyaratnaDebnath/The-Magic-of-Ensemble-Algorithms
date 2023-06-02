"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import entropy, information_gain, gini_index

np.random.seed(42)

class Node():
    def __init__(self):
        self.leaf = False
        self.child = dict()
        self.AtCat = False
        self.atid = None
        self.splitV = None
        self.val = None

class WeightedDecisionTree():
    def __init__(self, criterion, max_depth=None):
        self.criterion =  criterion 
        self.max_depth =  max_depth
        self.root      =  None


    def fill(self,x,y,newd,wt):
        newNode = Node()   #Creating a new Tree Node
        atid = -1
        splitV1 = None
        res = None

        if(y.dtype.name=="category"):
            df = pd.DataFrame({'y':y,'w':wt})
            gp = df.groupby(['y']).sum()
            map = np.unique(y)
            if(map.size==1):
                newNode.leaf = True
                newNode.Atcat = True
                newNode.val = map[0]
                return newNode
            if(self.max_depth!=None):
                if(self.max_depth==newd):
                    newNode.leaf = True
                    newNode.AtCat = True
                    newNode.val = gp['w'].idxmax()
                    return newNode
            if(x.shape[1]==0):
                newNode.leaf = True
                newNode.AtCat = True
                newNode.val = gp['w'].idxmax()
                return newNode
            
            for i in x:
                col =x[i]
                if(col.dtype.name=="category"):
                    res1 = None
                    if(self.criterion=="information_gain"):         
                        res1= information_gain(y,col,wt)
                    else:                                          
                        c = np.unique(col)
                        sum = 0
                        for j in c:
                            ycap = pd.Series([y[k] for k in range(y.size) if col[k]==j])
                            weight= pd.Series([wt[k] for k in range(wt.size) if col[k]==j])
                            sum =sum + ycap.size*gini_index(ycap,weight)
                        res1  = -1*(sum/(wt.size))
                    if(res!=None):
                        if(res<res1):
                            atid = i
                            res = res1
                            splitV1 = None
                    else:
                        atid = i
                        res = res1
                        splitV1 = None
               #####

                else:
                    col1 = col.sort_values()
                    for j in range(col1.size-1):
                        pos1 = col1.index[j]
                        pos2 = col1.index[j+1]
                        if(y[pos1]!=y[pos2]):
                            res1 = None
                            splitV = (col[pos1]+col[pos2])/2
                            
                            if(self.criterion=="information_gain"):                 # Criteria is Information Gain
                                newatr = pd.Series(col<=splitV)
                                res1 = information_gain(y,newatr,wt)
                            
                            else:                                                   # Criteria is Gini Index
                                y1 = pd.Series([y[k] for k in range(y.size) if col[k]<=splitV])
                                weight1 = pd.Series([wt[k] for k in range(wt.size) if col[k]<=splitV])
                                y2 = pd.Series([y[k] for k in range(y.size) if col[k]>splitV])
                                weight2 = pd.Series([wt[k] for k in range(wt.size) if col[k]>splitV])
                                res1 = y1.size*gini_index(y1,weight1) + y2.size*gini_index(y2,weight2)
                                res1 =  -1*(res1/np.sum(wt))
                            if(res!=None):
                                if(res<res1):
                                    atid = i
                                    res = res1
                                    splitV1 = splitV
                            else:
                                atid = i
                                res = res1
                                splitV1 = splitV  


             ##regression
        else:
            if(self.max_depth!=None):
                if(self.max_depth==newd):
                    newNode.leaf = True
                    newNode.val = y.mean()
                    return newNode
            if(y.size==1):
                newNode.leaf = True
                newNode.val = y[0]
                return newNode
            if(x.shape[1]==0):
                newNode.leaf = True
                newNode.val = y.mean()
                return newNode
            
            for i in x:
                col = x[i]
                if(col.dtype.name=="category"):
                    c = np.unique(col)
                    res1= 0
                    for j in c:
                        ycap = pd.Series([y[k] for k in range(y.size) if col[k]==j])
                        res1 =res1+ ycap.size*np.var(ycap)
                    if(res!=None):
                        if(res>res1):
                            res = res1
                            atid = i
                            splitV1 = None
                    else:
                        res = res1
                        atid = i
                        splitV1 = None

                ##Real Input Real Output
                else:
                    c1 = col.sort_values()
                    for j in range(y.size-1):
                        pos1 = c1.index[j]
                        pos2 = c1.index[j+1]
                        splitV = (col[pos1]+col[pos2])/2
                        y1 = pd.Series([y[k] for k in range(y.size) if col[k]<=splitV], dtype='float64')
                        y2 = pd.Series([y[k] for k in range(y.size) if col[k]>splitV], dtype='float64')
                        res1 = y1.size*np.var(y1) + y2.size*np.var(y2)
                        # c1 = y_sub1.mean()
                        # c2 = y_sub2.mean()
                        # measure = np.mean(np.square(y_sub1-c1) + np.square(y_sub2-c2))
                        if(res!=None):
                            if(res>res1):
                                atid = i
                                res = res1
                                splitV1= splitV
                        else:
                            atid = i
                            res = res1
                            splitV1= splitV

       #  split based
        if(splitV1==None): # when current Node is category based
            newNode.AtCat = True
            newNode.atid= atid
            c = np.unique(x[atid])
            for j in c:
                y1 = pd.Series([y[k] for k in range(y.size) if x[atid][k]==j], dtype=y.dtype)
                new_weight= pd.Series([wt[k] for k in range(wt.size) if x[atid][k]==j], dtype = wt.dtype)
                X1 = x[x[atid]==j].reset_index().drop(['index',atid],axis=1)
                newNode.child[j] = self.fill(X1, y1, newd+1,new_weight)
        # when current Node is category based        
        else:
            newNode.atid = atid
            newNode.splitV = splitV1
            y1 = pd.Series([y[k] for k in range(y.size) if x[atid][k]<=splitV1], dtype=y.dtype)
            X1= x[x[atid]<=splitV1].reset_index().drop(['index'],axis=1)
            new_weight1 = pd.Series([wt[k] for k in range(wt.size) if x[atid][k]<=splitV1], dtype=wt.dtype)
            y2 = pd.Series([y[k] for k in range(y.size) if x[atid][k]>splitV1], dtype=y.dtype)
            X2 = x[x[atid]>splitV1].reset_index().drop(['index'],axis=1)
            new_weight2 = pd.Series([wt[k] for k in range(wt.size) if x[atid][k]>splitV1], dtype=wt.dtype)
            newNode.child["lessThan"] = self.fill(X1, y1, newd+1,new_weight1)
            newNode.child["greaterThan"] = self.fill(X2, y2, newd+1,new_weight2)
        return newNode                       
        


    def fit(self, X: pd.DataFrame, y: pd.Series,wt=None) -> None:
        """
        Function to train and construct the decision tree
        """
        assert(y.size>0)
        assert(X.shape[0]==y.size)
        if wt is None:
            wt = pd.Series([1 for i in range(y.size)])
        self.root = self.fill(X,y,0,wt)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        ycap = list()                  
        for i in range(X.shape[0]):
            row = X.iloc[i,:]          
            h = self.root
            while(not h.leaf):                           
                if(h.AtCat):                       
                    h = h.child[row[h.atid]]
                else:                                      
                    if(row[h.atid]<=h.splitV):
                        h = h.child["lessThan"]
                    else:
                        h = h.child["greaterThan"]          
            ycap.append(h.val)                              
        ycap = pd.Series(ycap)
        return ycap

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        newhead = self.root
        result = self.Disp(newhead,0)
        print(result)
      
    def Disp(self, root, d):
        if(root.leaf):
            if(root.AtCat):
                return "Class "+str(root.val)
            else:
                return "Value "+str(root.val)

        out = ""
        if(root.AtCat):
            for i in root.child.keys():
                out =out+ "?("+str(root.atid)+" == "+str(i)+")\n" 
                out =out+ "\t"*(d+1)
                out =out+ str(self.Disp(root.child[i], d+1)).rstrip("\n") + "\n"
                out =out+ "\t"*(d)
            out = out.rstrip("\t")
        else:
            out  = out+"?("+str(root.atid)+" <= "+str(root.splitV)+")\n"
            out  = out+"\t"*(d+1)
            out  = out+"Y: " + str(self.Disp(root.child["lessThan"], d+1)).rstrip("\n") + "\n"
            out  = out+"\t"*(d+1)
            out  =out+ "N: " + str(self.Disp(root.child["greaterThan"], d+1)).rstrip("\n") + "\n"
        
        return out
               
