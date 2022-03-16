'''
Name: MCS.py
Desriptption: A class for implement modified Cuckoo Search algorithm 
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: For Pytorch 
Author: YesunHuang
Date: 2022-03-16 19:38:26
'''
### import everything
import math
from msilib.schema import Class
from sympy import GoldenRatio
import torch
from torch import nn
from torch.nn import functional as F

class MCS_optimizer:
    '''A class for implement MCS'''

    def __init__(self,costFunc,weightRange,**kwargs):
        '''
        name:__init__ 
        fuction: initialize the optimizer
        param {*costFunc}: cost function
        param {*weightRange}: function range
        param {***kargs}:
            param{*p_drop}: ratio of bad nest that to be dropped
            param{*nestNum}: number of maximum nests
            param{*maxGeneration}: maximum of Generation
            param{*maxLevyStepSize}: Levy step size
            param{*batchNum}: Num of Batches
        return {*}
        '''     
        self.costFunc=costFunc
        self.weightRange=weightRange
        if 'p_drop' in kwargs:
            self.p_drop=kwargs['p_drop']
        else:
            self.p_drop=0.75
        if 'nestNum' in kwargs:
            self.nestNum=kwargs['nestNum']
        else:
            self.nestNum=25
        if 'maxGeneration' in kwargs:
            self.maxGeneration=kwargs['maxGeneration']
        else:
            self.maxGeneration=100
        if 'maxLevyStepSize' in kwargs:
            self.maxLevyStepSize=kwargs['maxLevyStepSize']
        else:
            self.maxLevyStepSize=1
        if 'batchNum' in kwargs:
            self.batchNum=kwargs['batchNum']
        else:
            self.batchNum=1

    def __levyFlight(self, dimension,step, naive):
        '''
        name: __levyFlight
        fuction: helper function for implement two type of levy flight
        param {*dimension}: dimension of the space
        param {*step}: levy step
        param {*naive}: if using a naive flight or rigorous flight.
        return {*flight step}
        '''        
        if naive:
            distance=step*(torch.rand(dimension)-0.5*torch.ones(dimension))*torch.pow(torch.rand(dimension),-1.0/dimension)
            return distance*torch.ones(dimension)
        else:
            radius=2.0
            direction=torch.ones(dimension)
            while radius>1.0:
                direction=torch.rand(dimension)-0.5*torch.ones(dimension)
                radius=torch.sqrt(torch.sum(direction**2)).item()
            direction=direction/radius
            distance=step*torch.pow(torch.rand(1),-1.0/dimension)
            return distance*direction
