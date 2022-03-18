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
from cmath import sqrt
import abc
import math
import numpy as np
from sympy import GoldenRatio
import torch
from torch import nn
from torch.nn import functional as F

class standardCostFunc(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(X:torch.Tensor,Y:torch.Tensor,weight:torch.Tensor):
        pass

class MCS_optimizer:
    '''A class for implement MCS'''

    def __init__(self, netWeight:torch.Tensor, costFunc:standardCostFunc, dataIter, **kwargs):
        '''
        name:__init__ 
        fuction: initialize the optimizer
        param {*netWeight}: a vector of net Weight
        param {*costFunc}: class of cost function
        param {*dataIter}: iterator of data
        param {***kargs}:
            param{*p_drop}: ratio of bad nest that to be dropped
            param{*nestNum}: number of maximum nests
            param{*maxGeneration}: maximum of Generation
            param{*maxLevyStepSize}: Levy step size
        return {*MCS_optimizer}
        '''     
        self.costFunc=costFunc
        self.netWeight=netWeight
        self.dataIter=dataIter
        if 'p_drop' in kwargs:
            self.p_drop=kwargs['p_drop']
        else:
            self.p_drop=0.75
        if 'nestNum' in kwargs:
            self.nestNum=kwargs['nestNum']
        else:
            self.nestNum=25
        if 'maxLevyStepSize' in kwargs:
            self.maxLevyStepSize=kwargs['maxLevyStepSize']
        else:
            self.maxLevyStepSize=1.0
        self.currentGeneration=0
        self.__initialize()

    def __levy_flight(self, dimension,step, naive):
        '''
        name: __levyFlight
        fuction: helper function for implement two type of levy flight
        param {*dimension}: dimension of the space
        param {*step}: levy step
        param {*naive}: if using a naive flight or rigorous flight.
        return {*flight step}
        '''        
        if naive:
            distance=step*(2.0*torch.rand(dimension)-torch.ones(dimension))*torch.pow(torch.rand(dimension),-1.0/dimension)
            return distance*torch.ones(dimension)
        else:
            radius=2.0
            direction=torch.ones(dimension)
            while radius>1.0:
                direction=2*torch.rand(dimension)-torch.ones(dimension)
                radius=torch.sqrt(torch.sum(direction**2)).item()
            direction=direction/radius
            distance=step*torch.pow(torch.rand(1),-1.0/dimension)
            return distance*direction
    
    def __initialize(self):
        '''
        name: __initialize
        fuction: initialize all the nests
        return {*initial loss}
        '''            
        self.nestWeight=[]
        self.nestIndexAndCost=[]
        for X,Y in self.dataIter:
            cost=self.costFunc.evaluate(X,Y,self.netWeight)
            for i in range(0,self.nestNum):
                self.nestWeight.append(self.netWeight.clone())
                self.nestIndexAndCost.append((i,cost))
            break
        return cost
    
    def step(self,**kwarg):
        '''
        name: step
        fuction: update the nests
        param {***kwarg}:
            param{*isNaive}: whether using naive LevyFlight.
        return {*loss}
        '''     
        if 'isNaive' in kwarg:
            isNaive=kwarg['isNaive']
        else:
            isNaive=False
        self.currentGeneration+=1
        #calculate current levy step
        currentLevyStep=self.maxLevyStepSize/sqrt(self.currentGeneration)
        epochLoss=[]
        #iteration across all the batches
        for X,Y in self.dataIter:
            #sort all the nests by order of cost
            self.nestIndexAndCost.sort(key=lambda element:element[1])
            #update abandoned nests
            epochLoss.append(self.nestIndexAndCost[0][1])
            startAbandonIndex=self.nestNum-round(self.nestNum*self.p_drop)
            for i in range(startAbandonIndex,self.nestNum):
                deltaWeight=self.__levy_flight(self.nestWeight.shape[0],\
                                               currentLevyStep,\
                                               isNaive)
                self.nestWeight[self.nestIndexAndCost[i][0]].add_(deltaWeight)
                self.nestIndexAndCost[i][1]=self.costFunc.evaluate(\
                                            X,Y,\
                                            self.nestWeight[self.nestIndexAndCost[i][0]])
            #update top nests
            for i in range(0,startAbandonIndex):
                j=np.random.randint(0,startAbandonIndex,1)
                if self.nestWeight[self.nestIndexAndCost[j]].equal(\
                    self.nestWeight[self.nestIndexAndCost[i]]):
                    topLevyStep=self.maxLevyStepSize/(self.currentGeneration**2)
                    deltaWeight=self.__levy_flight(self.nestWeight.shape[0],\
                                               topLevyStep,\
                                               isNaive)
                    newWeight=self.nestWeight[self.nestIndexAndCost[j][0]].clone()
                    newWeight.add_(deltaWeight)
                    newCost=self.costFunc.evaluate(X,Y,newWeight)
                    k=np.random.randint(0,self.nestNum)
                    if newCost<self.nestIndexAndCost[k][1]:
                        self.nestWeight[self.nestIndexAndCost[k][0]]=newWeight.clone()
                        self.nestIndexAndCost[k][1]=newCost
                else:
                    deltaWeight=(self.nestWeight[self.nestIndexAndCost[i]]\
                            -self.nestWeight[self.nestIndexAndCost[j]])/(float(GoldenRatio))
                    if self.nestIndexAndCost[i]<self.nestIndexAndCost[j]:
                        newWeight=self.nestWeight[self.nestIndexAndCost[j][0]].clone()
                        newWeight.add_(deltaWeight)
                    else:
                        newWeight=self.nestWeight[self.nestIndexAndCost[i][0]].clone()
                        newWeight.add_(-deltaWeight)
                    newCost=self.costFunc.evaluate(X,Y,newWeight)
                    k=np.random.randint(0,self.nestNum)
                    if newCost<self.nestIndexAndCost[k][1]:
                        self.nestWeight[self.nestIndexAndCost[k][0]]=newWeight.clone()
                        self.nestIndexAndCost[k][1]=newCost
        self.nestIndexAndCost.sort(key=lambda element:element[1])
        topWeight=self.nestWeight[self.nestIndexAndCost[0][0]].clone()
        self.netWeight.add_(-self.netWeight+topWeight)
        return (self.nestIndexAndCost[0][1],np.mean(np.asarray(epochLoss)))                 