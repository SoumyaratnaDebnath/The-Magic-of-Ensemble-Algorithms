from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write her
    ycap =list(y_hat)
    ygt= list(y)
    i=0
    sum=0
    while i <(len(ycap)):
          if ycap[i]==ygt[i]:
            sum=sum+1
          i=i+1  
    return (sum/len(ycap))      

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    ycap =list(y_hat)
    ygt= list(y)
    i=0
    j=0
    sum1=0
    sum2=0
    while i<len(ycap):
         if ycap[i]==cls:
            sum2=sum2+1
         i=i+1   

    while j <(len(ycap)):
        if ycap[j]==cls and ycap[j]==ygt[j]:
            sum1=sum1+1
        j=j+1 


    return (sum1/sum2)  


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    ycap =list(y_hat)
    ygt= list(y)
    i=0
    j=0
    sum1=0
    sum2=0
    while i<len(ygt):
         if ygt[i]==cls:
            sum2=sum2+1
         i=i+1   

    while j <(len(ycap)):
        if ycap[j]==cls and ycap[j]==ygt[j]:
            sum1=sum1+1
        j=j+1 


    return (sum1/sum2)  

    


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    ycap =list(y_hat)
    ygt= list(y)
    N = len(ycap)
    sum=0
    for i in range(N):
        sum=sum+(ycap[i]-ygt[i])*(ycap[i]-ygt[i])

    return (sum/N)**(1/2)  


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    ycap =list(y_hat)
    ygt= list(y)
    N = len(ycap)
    sum=0
    for i in range(N):
        sum=(sum+abs(ycap[i]-ygt[i]))

    return (sum/N)