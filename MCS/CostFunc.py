'''
Name: TestCostFunc
Desriptption: Implement some of the cost funcs to test the MCS
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: For quantum RNN
Author: YesunHuang
Date: 2022-03-19 13:14:16
'''
### import everything
import torch
import abc
from MCS import StandardCostFunc
from torch import nn


class StandardSNN(metaclass=abc.ABCMeta):
    '''A abstract base class for standard sNN'''
    
    @abc.abstractmethod
    def call_with_weight(self,X:torch.Tensor,weight:tuple):
        '''A abstract method which forward with given weight'''
        pass

class GradFreeMSELoss(StandardCostFunc):
    '''A L2 cost function for simple NN'''

    def __init__(self,net:StandardSNN,**kwargs):
        '''
        name: __init__
        fuction: initialize the gradient free MSE loss
        param {*net}: the NN net attached to this cost function
        param {**kwargs}: the same with nn.MSE
        '''        
        self.net=net
        self.kwargs=kwargs
        self.loss=nn.MSELoss(**self.kwargs)
    
    def __call__(self,YHat,Y):
        '''
        name: __call__
        fuction: simply return the pytorch MSE
        param {*YHat}: NN output
        param {*Y}: real output
        return {*loss}
        '''    
        return self.loss(YHat,Y)

    @torch.no_grad()
    def evaluate(self,X: torch.Tensor, Y: torch.Tensor, weight: tuple):
        '''
        name: evaluate
        fuction: compute the cost or loss
        param {*X}: input for the nn
        param {*Y}: label output
        return {*cost}: current cost
        '''       
        if 'reduction' in self.kwargs:
            if self.kwargs['reduction']=='none':
                l=self.loss(self.net.call_with_weight(X,weight),Y).sum()
            else:
                l=self.loss(self.net.call_with_weight(X,weight),Y)
        return l.item()

class GradFreeCrossEntropyLoss(StandardCostFunc):
    '''A cross entropy cost function'''

    def __init__(self,net:StandardSNN,**kwargs):
        '''
        name: __init__
        fuction: initialize the gradient free MSE loss
        param {*net}: the NN net attached to this cost function
        param {**kwargs}: the same with nn.MSE
        '''        
        self.net=net
        self.kwargs=kwargs
        self.loss=nn.CrossEntropyLoss(**self.kwargs)
    
    def __call__(self,YHat,Y):
        '''
        name: __call__
        fuction: simply return the pytorch MSE
        param {*YHat}: NN output
        param {*Y}: real output
        return {*loss}
        '''    
        return self.loss(YHat,Y)

    @torch.no_grad()
    def evaluate(self,X: torch.Tensor, Y: torch.Tensor, weight: tuple):
        '''
        name: evaluate
        fuction: compute the cost or loss
        param {*X}: input for the nn
        param {*Y}: label output
        return {*cost}: current cost
        '''       
        if 'reduction' in self.kwargs:
            if self.kwargs['reduction']=='none':
                l=self.loss(self.net.call_with_weight(X,weight),Y).sum()
            else:
                l=self.loss(self.net.call_with_weight(X,weight),Y)
        return l.item()