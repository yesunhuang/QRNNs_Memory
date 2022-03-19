'''
Name: MCS.py
Desriptption: A class for implementing modified Cuckoo Search algorithm 
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
    def evaluate(X:torch.Tensor,Y:torch.Tensor,weight:tuple):
        pass

class MCS_optimizer:
    '''A class for implementing MCS'''

    def __init__(self, netWeight:tuple, costFunc:standardCostFunc, dataIter, **kwargs):
        '''
        name:__init__ 
        fuction: initialize the optimizer
        param {*netWeight}: a tuple of net Weight
        param {*costFunc}: class of cost function
        param {*dataIter}: iterator of data
        param {***kargs}:
            param{*p_drop}: ratio of bad nest that to be dropped
            param{*nestNum}: number of maximum nests
            param{*maxGeneration}: maximum of Generation
            param{*maxLevyStepSize}: Levy step size
            param{*randomInit}: use randn to initialize
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
        if 'randomInit' in kwargs:
            self.randInit=kwargs['randomInit']
        else:
            self.randInit=False
        self.currentGeneration=0
        self.__initialize()

    def levy_flight(self, dimension, step, naive):
        '''
        name: __levyFlight
        fuction: helper function for implement two types of levy flight
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
        '''            
        self.nestWeight=[]
        self.nestIndexAndCost=[]
        X,Y=next(iter(self.dataIter))
        if self.randomInit:
            for i in range(1,self.nestNum):
                newWeightTuple=()
                for weight in self.netWeight:
                    minWeight=torch.min(weight)
                    maxWeight=torch.max(weight)
                    newWeight=minWeight+(maxWeight-minWeight)*torch.randn(weight.shape)
                    newWeightTuple+=(newWeight.clone(),)
                self.nestWeight.append(newWeightTuple)
                cost=self.costFunc.evaluate(X,Y,newWeightTuple)
                self.nestIndexAndCost.append((i,cost))
        else:
            cost=self.costFunc.evaluate(X,Y,self.netWeight)
            for i in range(0,self.nestNum):
                newWeightTuple=()
                for weight in self.netWeight:
                    newWeightTuple+=(weight.clone(),)
                    self.nestWeight.append(newWeightTuple)
                    self.nestIndexAndCost.append((i,cost))
    
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
        getDeltaWeight=lambda weight,step:self.levy_flight(\
            weight.numel(),step,isNaive).reshape(weight.shape())
        #iteration across all the batches
        for X,Y in self.dataIter:
            #sort all the nests by order of cost
            self.nestIndexAndCost.sort(key=lambda element:element[1])
            #update abandoned nests
            epochLoss.append(self.nestIndexAndCost[0][1])
            startAbandonIndex=self.nestNum-round(self.nestNum*self.p_drop)
            for i in range(startAbandonIndex,self.nestNum):
                for weight in self.nestWeight[self.nestIndexAndCost[i]]:
                    deltaWeight=getDeltaWeight(weight,currentLevyStep)
                    weight.add_(deltaWeight)
                self.nestIndexAndCost[i][1]=self.costFunc.evaluate(X,Y,\
                    self.nestWeight[self.nestIndexAndCost[i][0]])
            #update top nests
            for i in range(0,startAbandonIndex):
                j=np.random.randint(0,startAbandonIndex,1)
                if i==j:
                    topLevyStep=self.maxLevyStepSize/(self.currentGeneration**2)
                    newWeightTuple=()
                    for weight in self.netWeight:
                        deltaWeight=getDeltaWeight(weight,topLevyStep)
                        newWeight=weight.clone()
                        newWeight.add_(deltaWeight)
                        newWeightTuple+=(newWeight,)
                    newCost=self.costFunc.evaluate(X,Y,newWeightTuple)
                    k=np.random.randint(0,self.nestNum)
                    if newCost<self.nestIndexAndCost[k][1]:
                        self.nestWeight[self.nestIndexAndCost[k][0]]=newWeightTuple
                        self.nestIndexAndCost[k][1]=newCost
                else:
                    #careful implement required
                    flag=self.nestIndexAndCost[i]<self.nestIndexAndCost[j]
                    weightTuple_i=self.nestWeight[self.nestIndexAndCost[i][0]]
                    weightTuple_j=self.nestWeight[self.nestIndexAndCost[j][0]]
                    newWeightTuple=()
                    for weightIndex in range(0,len(self.netWeight)):
                        deltaWeight=(weightTuple_i[weightIndex]\
                            -weightTuple_j[weightIndex])/(float(GoldenRatio))
                        if flag:
                            newWeight=weightTuple_j[weightIndex].clone()
                            newWeight.add_(deltaWeight)
                        else:
                            newWeight=weightTuple_i[weightIndex].clone()
                            newWeight.add_(-deltaWeight)
                        newWeightTuple+=(newWeight,)
                    newCost=self.costFunc.evaluate(X,Y,newWeightTuple)
                    k=np.random.randint(0,self.nestNum)
                    if newCost<self.nestIndexAndCost[k][1]:
                        self.nestWeight[self.nestIndexAndCost[k][0]]=newWeightTuple
                        self.nestIndexAndCost[k][1]=newCost
        self.nestIndexAndCost.sort(key=lambda element:element[1])
        #update netWeight
        for weightIndex in range(0,len(self.netWeight)):
            topWeight=self.nestWeight[self.nestIndexAndCost[0][0]][weightIndex]
            self.netWeight[weightIndex].add_(-self.netWeight[weightIndex]+topWeight)
        return (self.nestIndexAndCost[0][1],np.mean(np.asarray(epochLoss)))                 