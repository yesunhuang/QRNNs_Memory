'''
Name:HenonMapDataGen
Desriptption: It is used to generate the data of modified Henon Map
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: For quantum recurrrent neural networks
Author: YesunHuang
Date: 2022-03-26 20:45:29
'''

#import everything
import pandas as pd
import numpy as np
import torch
import os
from SequenceDataLoader import  SeqDataLoader

class HenonMapDataGen:
    '''Generate data of modified Henon Map'''

    def __init__(self, seed:list=[],\
                        n:int=1,a:float=1.4,b:float=0.3,\
                        heavyMem:bool=True,bound:bool=-1.2,\
                        savepath:str=os.getcwd()):
        '''
        name: __init__
        fuction: initialize the Henon map
        param {seed}: seed for generation
        param {n}: interval
        param {a}: Henon value a
        param {b}: Henon value b
        param {HeavyMem}: if using a heavy memory
        param {bound}: bound of the data
        param {savepath}: path to save the data
        '''   
        self.heavyMem=heavyMem
        self.interval=n
        if len(seed)==0:
            self.random_seed()
        else:
            self.__seed=seed
        if self.heavyMem:
            assert len(self.__seed)==n+1,'invalid seed!'
        else:
            assert len(self.__seed)==2*n,'invalid seed!'    
        self.paramA=a
        self.paramB=b
        self.HenonFunc=lambda X1,X0:1-self.paramA*X1*X1+self.paramB*X0
        self.savepath=savepath
        self.bound=bound
        self.__X=[]
        self.__Y=[]

    def __call__(self, size:int):
        '''
        name:__call__ 
        fuction: generate the Henon data
        param {size}: size of the data
        return {X,Y}: tuple of list in the form (X,Y)
        '''       
        self.clear_data()
        self.__X=self.__X+self.__seed
        self.__Y=self.__Y+[0.0]*self.interval
        if self.heavyMem:
            assert size>len(self.__seed), 'size not enough!'
            for i in range(self.interval,size):
                Y_next=self.HenonFunc(self.__X[i],self.__X[i-self.interval])
                if self.interval>1:
                    self.__Y.append(max(Y_next,self.bound))
                else:
                    self.__Y.append(Y_next)
                self.__X.append(self.__Y[i])
            self.__X.pop()
        else:
            assert size>len(self.__seed)+self.interval, 'size not enough'
            for i in range(self.interval,size):
                self.__Y.append(self.HenonFunc(self.__X[i],self.__X[i-self.interval]))
                if i+self.interval<size:
                    self.__X.append(self.__Y[i])
        return np.array(self.__X),np.array(self.__Y)

    def random_seed(self):
        '''
        name: random_seed
        function: random the seed
        return {seed}
        '''
        if self.heavyMem:
            self.__seed=[np.random.rand()*0.1 for i in range(self.interval+1)]
        else:
            self.__seed=[np.random.rand()*0.1 for i in range(2*self.interval)]   
        return self.__seed   

    @property
    def seed(self):
        '''
        name: seed
        function: get the seed
        return {seed}
        '''
        return self.__seed 

    def save_to_CSV(self,fileName:str):
        '''
        name: save_to_CSV
        function: save the data to csv file
        param {fileName}: name of the file
        '''
        path=os.path.join(self.savepath,fileName)
        data=pd.DataFrame({'X':self.__X,'Y':self.__Y})
        data.to_csv(path,index=False)
    
    def read_from_CSV(self,fileName:str):
        '''
        name: read_from_CSV
        function: read the data from csv file
        param {fileName}: name of the file
        '''
        path=os.path.join(self.savepath,fileName)
        data=pd.read_csv(path)
        self.__X=data['X'].values.tolist()
        self.__Y=data['Y'].values.tolist()

    @property
    def data_as_array(self):
        '''
        name: get_data
        function: get the data
        return {X,Y}: tuple of list in the form (X,Y)
        '''
        return np.array(self.__X),np.array(self.__Y)

    @property
    def data_as_tensor(self):
        '''
        name: get_data_as_tensor
        function: get the data as tensor
        return {X,Y}: tuple of tensor in the form (X,Y)
        '''
        return torch.tensor(self.__X),torch.tensor(self.__Y)

    def clear_data(self):
        '''
        name: clear_data
        function: clear the data
        '''
        self.__X=[]
        self.__Y=[]

    def get_data_iter(self,testSetRatio:float,numStep:int,\
                        batchSize:int=1,shuffle:bool=True):
        '''
        name: get_data_iter 
        fuction: get the data iter for nns
        param {testSetRatio}: the ratio of test set
        param {numStep}: number of step for a single nn data
        param {batchSize}: size of the mini batch
        param {shuffle}: if shuffling the data
        return {trainIter,testIter}
        '''        
        assert testSetRatio>0.0 & testSetRatio<1.0,'invalid testSetRatio!'
        assert numStep>self.interval, 'invalid numStep!'

        testStartIndex=int(len(self.__X)*(1-testSetRatio))
        data_train=(self.__X[:testStartIndex-1],self.__Y[:testStartIndex-1])
        data_test=(self.__X[testStartIndex:],self.__Y[testStartIndex:])
        trainIter=SeqDataLoader(data_train,numStep,self.interval,batchSize,shuffle)
        testIter=SeqDataLoader(data_test,numStep,self.interval,batchSize,shuffle)
        return trainIter,testIter
        
        


        
        


        
        
