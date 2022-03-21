# %% [markdown]
# # Test MCS with Linear Regression

# %% [markdown]
# This is a notebook for testing MCS with linear regression

# %% [markdown]
# ## Import everything

# %% [markdown]
# modify setting for pytorch

# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# %% [markdown]
# import torch and other standard modules

# %%
#%matplotlib inline
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# %% [markdown]
# import MCS and CostFunc

# %%
from Optimizers import MCSOptimizer
from CostFunc import GradFreeMSELoss
from CostFunc import StandardSNN

# %% [markdown]
# ## Linear Regression Testing

# %% [markdown]
# ### Generate training data

# %%
def synthetic_data(w, b, num_examples):  
    """y = Xw + b + noise"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# %%
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# %%
print('features:', features[0],'\nlabel:', labels[0])

# %%
fig, axes = plt.subplots(1,1,figsize=(4,3))
axes.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(),s=1.5);

# %% [markdown]
# ### Reading the data via torch API

# %%
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Generate a PyTorch data iterator"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# %%
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# %%
next(iter(data_iter))

# %% [markdown]
# ## Define a model

# %% [markdown]
# ### Function for initialize weight function

# %%
def getParams(num_inputs,num_outputs):
    w=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
    b=torch.zeros(1,requires_grad=True)
    return (w,b)

# %% [markdown]
# ### Linear Net Model

# %%
class LinearNet(StandardSNN):
    '''a standard net with linear regression'''
    def __init__(self,getParams,inputSize,numHiddens):
        self.params=getParams(inputSize,numHiddens)

    def __call__(self,X:torch.Tensor):
        return self.call_with_weight(X,self.params)

    def call_with_weight(self,X:torch.Tensor,weight:tuple):
        w,b=weight
        return torch.matmul(X,w)+b

# %%
net=LinearNet(getParams,2,1)
print(net.params)
rawParams=()
for param in net.params:
    rawParams+=(param.clone().detach().requires_grad_(True),)
print(rawParams)

# %% [markdown]
# ## Train a model

# %% [markdown]
# ### Set epochs

# %%
num_epochs=5

# %% [markdown]
# ### Loss function

# %%
lossFunc=GradFreeMSELoss(net)

# %% [markdown]
# ### MCS optimizer

# %%
mcs=MCSOptimizer(net.params,lossFunc,data_iter,\
                    maxLevyStepSize=[0.3,0.1],\
                    nestNum=8)

# %%
l_epochs_mcs=[float(lossFunc(net(features), labels))]
print(f'epoch 0, loss {l_epochs_mcs[0]:f}')
for epoch in range(num_epochs):
    mcs.step()
    with torch.no_grad():
        train_l = lossFunc(net(features), labels)
        l_epochs_mcs.append(float(train_l))
        print(f'epoch {epoch + 1}, loss {l_epochs_mcs[epoch+1]:f}')

# %%
w,b=net.params
print(w,b)
print(f'loss of w: {true_w - w.reshape(true_w.shape)}')
print(f'loss of b: {true_b - b}')

# %% [markdown]
# ### Standard sgd

# %%
net.params=rawParams
print(net.params)

# %%
trainer = torch.optim.SGD(net.params, lr=0.03)

# %%
lr = 0.03

# %%
l_epochs_sgd=[]
l_epochs_sgd=[float(lossFunc(net(features), labels))]
print(f'epoch 0, loss {l_epochs_sgd[0]:f}')
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = lossFunc(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    train_l = lossFunc(net(features), labels)
    l_epochs_sgd.append(float(train_l))
    print(f'epoch {epoch + 1}, loss {l_epochs_sgd[epoch+1]:f}')

# %%
w,b=net.params
print(w,b)
print(f'loss of w: {true_w - w.reshape(true_w.shape)}')
print(f'loss of b: {true_b - b}')

# %% [markdown]
# ## Visualize the Result

# %%
fig, axes = plt.subplots(1, 1, figsize=(4,3))

axes.semilogy(range(0,num_epochs+1),l_epochs_sgd,linestyle='-.',label='SGD')
axes.semilogy(range(0,num_epochs+1),l_epochs_mcs,linestyle='-',label='MCS')


axes.set_xlabel('Epochs')
axes.set_ylabel('Loss')

axes.xaxis.set_major_formatter(FormatStrFormatter('%d'))

axes.legend(loc=0,frameon=False);


