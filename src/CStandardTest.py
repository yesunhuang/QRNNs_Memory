'''
Name: CStandardTest
Desriptption: Performing the standard test for classical RNN
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: Experiment
Author: YesunHuang
Date: 2022-04-17 20:40:50
'''
#import all the things we need
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
from matplotlib.ticker import  FormatStrFormatter
import numpy as np
import pandas as pd
import torch
def transform(Xs):
        return [torch.squeeze(x) for x in Xs]
#Some constants
##File names
DATA_FILENAME='QExp1.csv'
NET_FILENAME='CExpFQC.pt'
TEST_DATA_FILENAME='QExp1Test.csv'
##Loss test configuration
TRIALS=20
TRIALS_STEP=1
TEST_TRIAN_DATA=False
##Prediction plot configuration
MULTI_PREFIX_SIZE=10
MULTI_TOTAL_SIZE=100
SINGLE_TOTAL_SIZE=100
DATA_SHIFT=0
SAVE_TEST_DATA=False
PLOT_ONLY=True


if __name__=='__main__':
    from DataGenerator.HenonMapDataGen import HenonMapDataGen
    from ClassicalModels.ClassicalSRNNs import ClassicalSRNN,SuportFunction
    from GradientFreeOptimizers.CostFunc import GradFreeMSELoss
    import GradientFreeOptimizers.Helpers as hp

    #Save path:
if __name__=='__main__':
    currentPath=os.getcwd()
    dataSavepath=os.path.join(currentPath,'data','HenonMap','Exp')
    netSavepath=os.path.join(currentPath,'TrainedNet','Exp')
    testSavepath=os.path.join(currentPath,'data','ModelTest','HenonMap','Exp')
    figSavepath=os.path.join(currentPath,'data','figures','STest')

if __name__=='__main__':
    # Data Iter
    ## Parameters
    testSetRatio=0.2
    numStep=10
    batchSize=16

if __name__=='__main__':   
    ## Read the data
    hmap=HenonMapDataGen(savepath=dataSavepath)
    hmap.read_from_CSV(DATA_FILENAME)
    ## Get the Iter
    trainIter,testIter=hmap.get_data_iter(testSetRatio,\
        numStep,batchSize,mask=0,shuffle=False,randomOffset=True)
    
    ## Print information
if __name__=='__main__':
    print(hmap)
    X,Y=next(iter(trainIter))
    print('Train Data Size:',len(trainIter))
    X,Y=next(iter(testIter))
    print('Test Data Size:',len(testIter))

    # Load the network
if  __name__=='__main__':
    netData=torch.load(os.path.join(netSavepath,NET_FILENAME))

    inputSize=netData['inputSize']
    outputSize=netData['outputSize']
    hiddenSize=netData['hiddenSize']
    
    inputRatio=netData['inputRatio']
    outputRatio=netData['outputRatio']
    initValue=netData['initValue']
    
    inactive=netData['inactive']
    rescale=netData['rescale']

    isTypical=netData['isTypical']

if __name__=='__main__':
    ## print parameters
    if isTypical:
        print('CRNN Type: Normal SRN')
    else:
        print('QRNN Type: Quantum Counterpart')
    print('Net Configuration:')
    print('-'*50)
    print('Total hidden units:',hiddenSize)
    print('Input Ratio:\n',inputRatio)
    print('Output Ratio:\n',outputRatio)
    print('Initial State Value:\n',initValue)
    print('-'*50)
    

if __name__=='__main__':
    ## Get neccesary functions
    srnnTestSup=SuportFunction()
    #transform=lambda Xs:[torch.squeeze(x) for x in Xs]
    init_rnn_state=srnnTestSup.get_init_state_fun(initStateValue=initValue)
    get_params=srnnTestSup.get_get_params_fun(inputRatio=inputRatio,\
                                                outputRatio=outputRatio,\
                                                rescale=rescale,\
                                                inactive=inactive)
    rnn=srnnTestSup.get_forward_fn_fun(isTypical=isTypical)
    predict_fun=srnnTestSup.get_predict_fun(outputTransoform=transform)

    net=ClassicalSRNN(inputSize,hiddenSize,outputSize,get_params,init_rnn_state,rnn)

if  __name__=='__main__':
    net.params=netData['NetParams']
    net.constants=netData['NetConstants']

## Loss
if __name__=='__main__' and not PLOT_ONLY:
    ## Loss function
    lossFunc=GradFreeMSELoss(net)
    l_epochs=netData['Loss']
    print(f'Saved Train Loss: {l_epochs[-1][0]:f}')
    print(f'Saved Test Loss: {l_epochs[-1][1]:f}')

    timer=hp.Timer()
    train_loss=[]
    test_loss=[]
    for i in range(TRIALS):
        test_loss.append(SuportFunction.evaluate_accuracy(net, testIter, lossFunc, True))
        if TEST_TRIAN_DATA:
            train_loss.append(SuportFunction.evaluate_accuracy(net, trainIter, lossFunc, True))
        if (i+1)%TRIALS_STEP==0:
            timeCost=timer.stop()
            print('-'*50)
            print(f'Trial {i+1:d}/{TRIALS:d}')
            print(f'Test Loss: {test_loss[-1]:f}')
            print(f'Time Cost: {timeCost:f}s')
            timer.start()
    print('-'*50)
    print(f'Average Test Loss: {np.mean(test_loss):f}')
    print(f'Test Loss Variance: {np.var(test_loss):f}')


if __name__=='__main__' and SAVE_TEST_DATA and not PLOT_ONLY:
    try:
        testDf=pd.read_csv(os.path.join(testSavepath,TEST_DATA_FILENAME))
    except:
        testDf=pd.DataFrame(columns=['Name','Trial',\
            'AvgTestLoss','VarTestLoss','MinTestLoss','MaxTestLoss','SavedTrainLoss'])
    if NET_FILENAME in testDf['Name'].values:
        testDf.loc[testDf['Name']==NET_FILENAME]=[NET_FILENAME,TRIALS,\
            np.mean(test_loss),np.var(test_loss),np.min(test_loss),np.max(test_loss),l_epochs[-1][0]]
    else:
        testDf=testDf.append({'Name':NET_FILENAME,'Trial':TRIALS,\
            'AvgTestLoss':np.mean(test_loss),'VarTestLoss':np.var(test_loss),\
            'MinTestLoss':np.min(test_loss),'MaxTestLoss':np.max(test_loss),\
            'SavedTrainLoss':l_epochs[-1][0]},ignore_index=True)
    testDf.to_csv(os.path.join(testSavepath,TEST_DATA_FILENAME),index=False)

if  __name__=='__main__':
    figRootName=NET_FILENAME[:-3]
if  __name__=='__main__':
    # Prediction
    ## One-step prediction
    testShift=int(len(hmap)*(1-testSetRatio))+DATA_SHIFT
    preX,preY=hmap.data_as_tensor
    preX,preY=torch.unsqueeze(preX[testShift:testShift+SINGLE_TOTAL_SIZE],-1),\
        torch.unsqueeze(preY[testShift:testShift+SINGLE_TOTAL_SIZE],-1)
    preY=[y for y in torch.cat((preX[:hmap.interval+1],preY[hmap.interval:]),dim=0)]
    preX=torch.unsqueeze(preX,-1)
    YHat=predict_fun(preX,net,multiPred=False)

    fig,axes=plt.subplots(1,1,figsize=(4,3))
    axes.set_title('Single-Step Prediction')
    axes.set_ylim(-2,2)
    axes.plot(torch.linspace(1,len(preY),len(preY)),preY,label=r'y_t')
    axes.plot(torch.linspace(1,len(preY),len(preY)),YHat,label=r'$\hat{y}_t$')
    axes.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes.set_xlabel(r'$t$')
    axes.set_ylabel(r'$y_t$')
    axes.legend()
    plt.show()
    figName=figRootName+'_SP_T'
    fig.savefig(os.path.join(figSavepath,figName+'.svg'),dpi=600,format='svg',bbox_inches='tight')
    fig.savefig(os.path.join(figSavepath,figName+'.pdf'),dpi=600,format='pdf',bbox_inches='tight')

if __name__=='__main__':
    fig,axes=plt.subplots(1,1,figsize=(4,3))
    axes.set_title('Phase Diagram of single-Step Prediction')
    axes.set_ylim(-2,2)
    yHatList=[Y.item() for Y in YHat]
    yPreList=[Y.item() for Y in preY]
    axes.set_xlabel(r'$x_t$')
    axes.set_ylabel(r'$y_t$')
    axes.scatter(yHatList[MULTI_PREFIX_SIZE:-1],yHatList[MULTI_PREFIX_SIZE+1:],s=2,label=r'$\hat{y}$')
    axes.scatter(yPreList[MULTI_PREFIX_SIZE:-1],yPreList[MULTI_PREFIX_SIZE+1:],s=2,label=r'$y$')
    axes.legend()
    plt.show()
    figName=figRootName+'_SP_P'
    fig.savefig(os.path.join(figSavepath,figName+'.svg'),dpi=600,format='svg',bbox_inches='tight')
    fig.savefig(os.path.join(figSavepath,figName+'.pdf'),dpi=600,format='pdf',bbox_inches='tight')

if __name__=='__main__':
    ## Multi Step Prediction
    testShift=int(len(hmap)*(1-testSetRatio))+DATA_SHIFT
    preX,preY=hmap.data_as_tensor
    preX,preY=torch.unsqueeze(preX[testShift:testShift+MULTI_PREFIX_SIZE],-1),\
        torch.unsqueeze(preY[testShift:testShift+MULTI_TOTAL_SIZE-1],-1)
    preY=[y for y in torch.cat((preX[:hmap.interval+1],preY[hmap.interval:]),dim=0)]
    preX=torch.unsqueeze(preX,-1)
    YHat=predict_fun(preX,net,numPreds=MULTI_TOTAL_SIZE-MULTI_PREFIX_SIZE)

    fig,axes=plt.subplots(1,1,figsize=(4,3))
    axes.set_title('Multi-Step Prediction')
    axes.set_ylim(-2,2)
    axes.plot(torch.linspace(1,len(preY),len(preY)),preY,label=r'y_t')
    axes.plot(torch.linspace(1,len(preY),len(preY)),YHat,label=r'$\hat{y}_t$')
    axes.vlines([MULTI_PREFIX_SIZE],ymin=-2,ymax=2,linestyles='dashed',label='Prediction')
    axes.set_xlabel(r'$t$')
    axes.set_ylabel(r'$y_t$')
    axes.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes.legend()
    plt.show()
    figName=figRootName+'_MP_T'
    fig.savefig(os.path.join(figSavepath,figName+'.svg'),dpi=600,format='svg',bbox_inches='tight')
    fig.savefig(os.path.join(figSavepath,figName+'.pdf'),dpi=600,format='pdf',bbox_inches='tight')

if __name__=='__main__':
    fig,axes=plt.subplots(1,1,figsize=(4,3))
    axes.set_title('Phase Diagram of Multi-Step Prediction')
    axes.set_ylim(-2,2)
    yHatList=[Y.item() for Y in YHat]
    yPreList=[Y.item() for Y in preY]
    axes.set_xlabel(r'$x_t$')
    axes.set_ylabel(r'$y_t$')
    axes.scatter(yHatList[MULTI_PREFIX_SIZE:-1],yHatList[MULTI_PREFIX_SIZE+1:],s=2,label=r'$\hat{y}$')
    axes.scatter(yPreList[MULTI_PREFIX_SIZE:-1],yPreList[MULTI_PREFIX_SIZE+1:],s=2,label=r'$y$')
    axes.legend()
    plt.show()
    figName=figRootName+'_MP_P'
    fig.savefig(os.path.join(figSavepath,figName+'.svg'),dpi=600,format='svg',bbox_inches='tight')
    fig.savefig(os.path.join(figSavepath,figName+'.pdf'),dpi=600,format='pdf',bbox_inches='tight')


