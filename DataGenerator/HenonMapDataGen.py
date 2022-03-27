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
from SequenceDataLoader import SequenceDataLoader

class HenonMapDataGen:
    '''Generate data of modified Henon Map'''

    def __init__(self, seed:list,\
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
        if heavyMem:
            assert len(seed)==n+1,'invalid seed!'
        else:
            assert len(seed)==2*n,'invalid seed!'    
        self.seed=seed
        self.interval=n
        self.paramA=a
        self.paramB=b
        self.HenonFunc=lambda X1,X0:1-self.paramA*X1*X1+self.paramB*X0
        self.HeavyMem=heavyMem
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
        self.__X=self.__X+self.seed
        self.__Y=self.__Y+[0.0]*self.interval
        if self.HeavyMem:
            assert size>len(self.seed), 'size not enough!'
            for i in range(self.interval,size):
                Y_next=self.HenonFunc(self.__X[i],self.__X[i-self.interval])
                if self.interval>1:
                    self.__Y.append(max(Y_next,self.bound))
                else:
                    self.__Y.append(Y_next)
                self.__X.append(self.__Y[i])
            self.__X.pop()
        else:
            assert size>len(self.seed)+self.interval, 'size not enough'
            for i in range(self.interval,size):
                self.__Y.append(self.HenonFunc(self.__X[i],self.__X[i-self.interval]))
                if i+self.interval<size:
                    self.__X.append(self.__Y[i])
        return np.array(self.__X),np.array(self.__Y)

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

    def get_data(self):
        '''
        name: get_data
        function: get the data
        return {X,Y}: tuple of list in the form (X,Y)
        '''
        return np.array(self.__X),np.array(self.__Y)

    def get_data_as_tensor(self):
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



        
        
