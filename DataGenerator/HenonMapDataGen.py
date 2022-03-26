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
from asyncio.proactor_events import _ProactorBaseWritePipeTransport
import pandas as pd
import numpy as np
import os
import torch

class HenonMapDataGen:
    '''Generate data of modified Henon Map'''

    def __init__(self, seed:list,\
                        n:int=1,a:float=1.4,b:float=0.3,\
                        HeavyMem:bool=True,savepath:str=os.getcwd()):
        '''
        name: __init__
        fuction: initialize the Henon map
        param {seed}: seed for generation
        param {n}: interval
        param {a}: Henon value a
        param {b}: Henon value b
        param {HeavyMem}: if using a heavy memory
        '''   
        if HeavyMem:
            assert len(seed)==n+1,'invalid seed!'
        else:
            assert len(seed)==2*n,'invalid seed!'    
        self.seed=seed
        self.interval=n
        self.paramA=a
        self.paramB=b
        self.HenonFunc=lambda X1,X0:1-1.4*X1*X1+0.3*X0
        self.HeavyMem=HeavyMem
        self.savepath=savepath
        self.X=[]
        self.Y=[]

    def __call__(self, size:int):
        '''
        name:__call__ 
        fuction: generate the Henon data
        param {size}: size of the data
        return {X,Y}: tuple of list in the form (X,Y)
        '''        
        self.X=self.X+self.seed
        if self.HeavyMem:
            assert size>len(self.seed), 'size not enough!'
            self.Y=[0.0]*self.interval
            for i in range(self.interval,size):
                self.Y.append(self.HenonFunc(self.X[i],self.X[i-self.interval]))
                self.X.append(self.Y[i])
        else:
            assert size>len(self.seed)+self.interval, 'size not enough'
            self.Y=[0.0]*(2*self.interval)
            for i in range(self.interval,size-self.interval):
                self.Y.append(self.HenonFunc(self.X[i],self.X[i-self.interval]))
                self.X.append(self.Y[i])

        return self.X[:-1],self.Y

    def save_to_CSV(self,fileName:str):
        '''
        name: save_to_CSV
        function: save the data to csv file
        param {fileName}: name of the file
        '''
        path=os.path.join(self.savepath,fileName)
        data=pd.DataFrame({'X':self.X,'Y':self.Y})
        data.to_csv(path,index=False)
    
    def read_from_CSV(self,fileName:str):
        '''
        name: read_from_CSV
        function: read the data from csv file
        param {fileName}: name of the file
        '''
        path=os.path.join(self.savepath,fileName)
        data=pd.read_csv(path)
        self.X=data['X'].values
        self.Y=data['Y'].values



        
        
