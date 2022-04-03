'''
Name: SequenceDataLoader
Desriptption: Load the sequence data
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: For quantum recurrrent neural networks
Author: YesunHuang
Date: 2022-03-27 23:39:16
'''

#import everything
import random
import torch

class SeqDataLoader:
    '''A class for loading the training data'''

    def __init__(self, data:tuple,\
                numSteps, maskSteps:int=0,\
                batchSize:int=1, shuffle:bool=True):
        '''
        name: __init__ 
        fuction: initialize the data loader 
        param {data}: the raw data tuple (X,Y)
        param {numSteps}: the number of sample steps
        param {maskSteps}: initial masked output steps
        param {batchSize}: the batch size
        param {shuffle}: whether to shuffle the data
        '''
        self.X,self.Y = data
        self.numSteps = numSteps
        self.maskSteps = maskSteps
        self.batchSize = batchSize
        self.dataSize=0
        if shuffle:
            self.dataIterFn=self.__seq_data_iter_random
        else:
            self.dataIterFn=self.__seq_data_iter_sequential
    
    def __iter__(self):
        '''
        name: __iter__
        function: generate the data iterator
        return: the data iterator
        '''
        return self.dataIterFn()
        
    def __seq_data_iter_random(self):
        '''
        name: __seq_data_iter_random
        function: generate the random sequence data iterator
        return: the random sequence data iterator
        '''
        #random offset
        offset=random.randint(0,self.numSteps)
        Xs,Ys=self.X[offset:],self.Y[offset:]
        #number of sequences
        numSeqs=len(Xs)//self.numSteps
        self.dataSize=numSeqs
        #random initial index
        initialIndices=list(range(0,numSeqs*self.numSteps,self.numSteps))
        random.shuffle(initialIndices)

        def get_data(pos,ifMask):
            '''
            name: get_data
            function: get the data
            param {pos}: the position
            param {ifMask}: if masking the data
            return: the data
            '''
            if ifMask:
                Ys[pos:pos+self.maskSteps]=[0.0]*self.maskSteps
                return Ys[pos:pos+self.numSteps]
            return Xs[pos:pos+self.numSteps]

        for i in range(0,self.batchSize*self.numSteps,self.numSteps):
            initialIndicesPerBatch=initialIndices[i:i+self.batchSize]
            XsPerBatch=[get_data(j,False) for j in initialIndicesPerBatch]
            YsPerBatch=[get_data(j,True) for j in initialIndicesPerBatch]
            XsPerBatch=torch.tensor(XsPerBatch,dtype=torch.float32)
            YsPerBatch=torch.tensor(YsPerBatch,dtype=torch.float32)
            if len(XsPerBatch.shape)<3:
                XsPerBatch=torch.unsqueeze(XsPerBatch,dim=-1)
            if len(YsPerBatch.shape)<3:
                YsPerBatch=torch.unsqueeze(YsPerBatch,dim=-1)
            yield XsPerBatch,YsPerBatch

    def __seq_data_iter_sequential(self):
        '''
        name: __seq_data_iter_sequential
        function: generate the sequential sequence data iterator
        return: the sequential sequence data iterator
        '''
        offset=random.randint(0,self.numSteps)
        numData=(len(self.X)-offset)//self.batchSize*self.batchSize
        Xs=torch.tensor(self.X[offset:offset+numData],dtype=torch.float32)
        Ys=torch.tensor(self.Y[offset:offset+numData],dtype=torch.float32)
        Xs,Ys=Xs.reshape(self.batchSize,-1),Ys.reshape(self.batchSize,-1)
        numBatches=Xs.shape[1]//self.numSteps
        self.dataSize=numBatches*self.batchSize
        for i in range(0,self.numSteps*numBatches,self.numSteps):
            XsPerBatch=Xs[:,i:i+self.numSteps]
            YsPerBatch=Ys[:,i:i+self.numSteps]
            YsPerBatch[:,:self.maskSteps]=0
            if len(XsPerBatch.shape)<3:
                XsPerBatch=torch.unsqueeze(XsPerBatch,dim=-1)
            if len(YsPerBatch.shape)<3:
                YsPerBatch=torch.unsqueeze(YsPerBatch,dim=-1)
            yield XsPerBatch,YsPerBatch
        
    def __len__(self):
        '''
        name: __len__
        function: get the data size
        return: the data size
        '''
        return self.dataSize
