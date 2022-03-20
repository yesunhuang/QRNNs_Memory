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
import abc
import torch
import numpy as np
from sympy import GoldenRatio
from GradFreeOptimizer import StandardGradFreeOptimizer


class StandardCostFunc(metaclass=abc.ABCMeta):
    '''A abstract base class for cost function'''
    
    @abc.abstractmethod
    def evaluate(self,X:torch.Tensor,Y:torch.Tensor,weight:tuple):
        pass

class MCSOptimizer(StandardGradFreeOptimizer):
    '''A class for implementing MCS'''

    def __init__(self, netWeight:tuple, costFunc:StandardCostFunc, dataIter, **kwargs):
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
            param{*constantStep}: use constant levy step
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
        if 'constantStep' in kwargs:
            self.constantStep=kwargs['constantStep']
        else:
            self.constantStep=False
        self.currentGeneration=0
        self.__initialize()

    @torch.no_grad()
    def __levy_flight(self,dimension, step, naive):
        '''
        name: __levyFlight
        fuction: helper function for implement two types of levy flight
        param {*dimension}: dimension of the space
        param {*step}: levy step
        param {*naive}: if using a naive flight or rigorous flight.
        return {*flight step}
        '''        
        direction=2*torch.rand(dimension)-torch.ones(dimension)
        radius=torch.sqrt(torch.sum(direction**2)).item()
        if not naive:
            while radius>1.0:
                direction[:]=2*torch.rand(dimension)-torch.ones(dimension)
                radius=torch.sqrt(torch.sum(direction**2)).item()
        direction[:]=direction/radius
        distance=step*torch.pow(torch.rand(1),-1.0/dimension)
        return distance*direction
    
    @torch.no_grad()
    def __initialize(self):
        '''
        name: __initialize
        fuction: initialize all the nests
        '''            
        self.nestWeight=[]
        self.nestIndexAndCost=[]
        X,Y=next(iter(self.dataIter))
        if self.randInit:
            for i in range(0,self.nestNum):
                newWeightTuple=()
                for weight in self.netWeight:
                    minWeight=torch.min(weight)
                    maxWeight=torch.max(weight)
                    newWeight=minWeight+(maxWeight-minWeight)*torch.randn(weight.shape)
                    newWeightTuple+=(newWeight.clone(),)
                self.nestWeight.append(newWeightTuple)
                cost=self.costFunc.evaluate(X,Y,newWeightTuple)
                self.nestIndexAndCost.append([i,cost])
        else:
            cost=self.costFunc.evaluate(X,Y,self.netWeight)
            for i in range(0,self.nestNum):
                newWeightTuple=()
                for weight in self.netWeight:
                    newWeightTuple+=(weight.clone(),)
                self.nestWeight.append(newWeightTuple)
                self.nestIndexAndCost.append([i,cost])
    
    @torch.no_grad()
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
        if not self.constantStep:
            currentLevyStep=self.maxLevyStepSize/np.sqrt(self.currentGeneration)
        else:
            currentLevyStep=self.maxLevyStepSize
        epochLoss=[]
        getDeltaWeight=lambda weight,step:self.__levy_flight(\
            weight.numel(),step,isNaive).reshape(weight.shape)
        #iteration across all the batches
        for X,Y in self.dataIter:
            #sort all the nests by order of cost
            self.nestIndexAndCost.sort(key=lambda element:element[1])
            #update abandoned nests
            epochLoss.append(self.nestIndexAndCost[0][1])
            startAbandonIndex=self.nestNum-round(self.nestNum*self.p_drop)
            for i in range(startAbandonIndex,self.nestNum):
                for weight in self.nestWeight[self.nestIndexAndCost[i][0]]:
                    deltaWeight=getDeltaWeight(weight,currentLevyStep)
                    weight.add_(deltaWeight)
                self.nestIndexAndCost[i][1]=self.costFunc.evaluate(X,Y,\
                    self.nestWeight[self.nestIndexAndCost[i][0]])
            #update top nests
            for i in range(0,startAbandonIndex):
                j=np.random.randint(0,startAbandonIndex)
                if i==j:
                    if not self.constantStep:
                        topLevyStep=self.maxLevyStepSize/(self.currentGeneration**2)
                    else:
                        topLevyStep=self.maxLevyStepSize
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
                    flag=self.nestIndexAndCost[i][1]<self.nestIndexAndCost[j][1]
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

    @torch.no_grad()
    def zero_grad(self):
        '''
        name: zero_grad
        fuction: clear out all the grad
        '''
        for weight in self.netWeight:
            weight.grad.zero_()
        for weightTuple in self.nestWeight:
            for weight in weightTuple:
                weight.grad.zero_()