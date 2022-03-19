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
from functools import reduce
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

class SNNCostFuncL2(StandardCostFunc):
    '''A L2 cost function for simple NN'''
    def __init__(self,net:StandardSNN):
        self.net=net
        self.loss=nn.MSELoss(reduction='none')
    
    def __call__(self,YHat,Y):

        return self.loss(YHat,Y)

    def evaluate(self,X: torch.Tensor, Y: torch.Tensor, weight: tuple):
        '''
        name: evaluate
        fuction: compute the cost or loss
        param {*X}: input for the nn
        param {*Y}: label output
        return {*cost}: current cost
        '''        
        l=self.loss(self.net.call_with_weight(X,weight),Y).sum()
        return l.item()