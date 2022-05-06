
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import torch
from DataGenerator.HenonMapDataGen import HenonMapDataGen
#Some constants
GENERATE_DATA=False
SAVE_TEST_DATA=True
FILE_NAME='QExp1.csv'
TEST_DATA_FILENAME='QExp1Test.csv'
NET_FILENAME='Baseline'
INTERVAL=1
OFFSET=1
TIME_PLOT_SIZE=100
PHASE_PLOT_SIZE=500
TRIALS=20

#set the save path
currentPath=os.getcwd()
dataSavepath=os.path.join(currentPath,'data','HenonMap','Exp')
figSavepath=os.path.join(currentPath,'data','figures')
testSavepath=os.path.join(currentPath,'data','ModelTest','HenonMap','Exp')

# Generate Data
if GENERATE_DATA:
    hmap=HenonMapDataGen(savepath=dataSavepath,n=INTERVAL,heavyMem=False)
    hmap(1000)
    hmap.save_to_CSV(FILE_NAME)

# Read the data
hmap=HenonMapDataGen(savepath=dataSavepath)
hmap.read_from_CSV(FILE_NAME)
print(hmap)

# Get the Iter
testSetRatio=0.2
numStep=10
batchSize=16
trainIter,testIter=hmap.get_data_iter(testSetRatio,\
        numStep,batchSize,mask=0,shuffle=False,randomOffset=True)

#Baseline
test_loss=[]
test_mean=[]
for i in range(TRIALS):
    batchMean=[]
    batchVar=[]
    for _,Y in testIter:
        mean=torch.mean(Y)
        var=torch.var(Y)
        batchMean.append(mean.item())
        batchVar.append(var.item())
    test_loss.append(np.mean(batchVar))
    test_mean.append(np.mean(batchMean))
    print(f'Trial {i+1:d}/{TRIALS:d}')
    print(f'Test Loss: {test_loss[-1]:f}')
print('-'*50)
print(f'Average Test Loss: {np.mean(test_loss):f}')
print(f'Test Loss Variance: {np.var(test_loss):f}')

if __name__=='__main__' and SAVE_TEST_DATA:
    try:
        testDf=pd.read_csv(os.path.join(testSavepath,TEST_DATA_FILENAME))
    except:
        testDf=pd.DataFrame(columns=['Name','Trial',\
            'AvgTestLoss','VarTestLoss','MinTestLoss','MaxTestLoss','SavedTrainLoss'])
    if NET_FILENAME in testDf['Name'].values:
        testDf.loc[testDf['Name']==NET_FILENAME]=[NET_FILENAME,TRIALS,\
            np.mean(test_loss),np.var(test_loss),np.min(test_loss),np.max(test_loss),np.mean(test_mean)]
    else:
        testDf=testDf.append({'Name':NET_FILENAME,'Trial':TRIALS,\
            'AvgTestLoss':np.mean(test_loss),'VarTestLoss':np.var(test_loss),\
            'MinTestLoss':np.min(test_loss),'MaxTestLoss':np.max(test_loss),\
            'SavedTrainLoss':np.mean(test_mean)},ignore_index=True)
    testDf.to_csv(os.path.join(testSavepath,TEST_DATA_FILENAME),index=False)


#Plot the data
X,Y=hmap.data_as_array

#Plot the time series
Yt=Y[OFFSET:OFFSET+TIME_PLOT_SIZE]
fig,axes=plt.subplots(1,1,figsize=(4,3))
axes.plot(np.linspace(1,len(Yt),len(Yt)),Yt)
axes.set_ylabel(r'$y_t$')
axes.set_xlabel(r'$t$')
axes.xaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.show()
fig.savefig(os.path.join(figSavepath,'HenonT.svg'),dpi=600,format='svg',bbox_inches='tight')
fig.savefig(os.path.join(figSavepath,'HenonT.pdf'),dpi=600,format='pdf',bbox_inches='tight')

#Plot the phase space
Xp=X[OFFSET:OFFSET+PHASE_PLOT_SIZE]
Yp=Y[OFFSET:OFFSET+PHASE_PLOT_SIZE]
fig,axes=plt.subplots(1,1,figsize=(4,3))
axes.scatter(Xp,Yp,s=2.0)
axes.set_ylabel(r'$y_t$')
axes.set_xlabel(r'$x_t$')
plt.show()
fig.savefig(os.path.join(figSavepath,'HenonP.svg'),dpi=600,format='svg',bbox_inches='tight')
fig.savefig(os.path.join(figSavepath,'HenonP.pdf'),dpi=600,format='pdf',bbox_inches='tight')



