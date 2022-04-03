'''
Name: 
Desriptption: 
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: 
Author: YesunHuang
Date: 2022-04-03 21:18:35
'''
# %% [markdown]
# # Test the classical SRNN

# %% [markdown]
# This is a notebook for testing the classical SRNN.

# %% [markdown]
# ## Import everything

# %% [markdown]
# Modify setting for pytorch

# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
currentPath=os.getcwd()

# %% [markdown]
# Import matplotlib and others

# %%
import matplotlib.pyplot as plt
import torch

# %% [markdown]
# Import the classical SRNN and others

# %%
#Modify path for the notebooks
#currentPath=os.path.join(currentPath,'..')
currentPath=os.path.join(currentPath,'src')
os.chdir(currentPath)
import sys
sys.path.append(currentPath)

# %%
from DataGenerator.HenonMapDataGen import HenonMapDataGen
from ClassicalModels.ClassicalSRNNs import ClassicalSRNN
from ClassicalModels.ClassicalSRNNs import SuportFunction
from GradientFreeOptimizers.CostFunc import GradFreeMSELoss
import GradientFreeOptimizers.Helpers as hp

# %% [markdown]
# ## Test One

# %% [markdown]
# ### Get the data

# %% [markdown]
# Set save path

# %%
savepath=os.path.join(currentPath,'..\data\HenonMap\Test')
filename='HenonMapTest1.csv'

# %% [markdown]
# Read the data

# %%
hmap=HenonMapDataGen(savepath=savepath)
hmap.read_from_CSV(filename)

# %%
print(hmap)

# %% [markdown]
# Generate the data iter

# %%
testSetRatio=0.2
numStep=5
batchSize=4

# %%
trainIter,testIter=hmap.get_data_iter(testSetRatio,numStep,batchSize,mask=0,shuffle=False)

# %%
X,Y=next(iter(trainIter))
print('Train Data Size:',len(trainIter))
print('X=',X)
print('Y=',Y)

# %%
X,Y=next(iter(testIter))
print('Train Data Size:',len(testIter))
print('X=',X)
print('Y=',Y)

# %% [markdown]
# ### Define the SRNN

# %% [markdown]
# #### Get neccesary functions

# %%
srnnTest1Sup=SuportFunction()

# %%
init_rnn_state=srnnTest1Sup.get_init_state_fun()
get_params=srnnTest1Sup.get_get_params_fun()
rnn=srnnTest1Sup.get_forward_fn_fun()
predict_fun=srnnTest1Sup.get_predict_fun()

# %% [markdown]
# #### Create the SRNN

# %% [markdown]
# Parameters

# %%
inputSize=outputSize=1
hiddenSize=6

# %%
net=ClassicalSRNN(inputSize,hiddenSize,outputSize,get_params,init_rnn_state,rnn)

# %%
state=net.begin_state(batchSize)
Y,newState=net(X,state)
Y.shape, len(newState), newState[0].shape

# %%



