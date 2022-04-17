'''
Name: QExp1
Desriptption: Experiment One 
Email: yesunhuang@mail.ustc.edu.cn
OpenSource: https://github.com/yesunhuang
Msg: Experiment One
Author: YesunHuang
Date: 2022-04-17 20:40:50
'''
#import all the things we need
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import torch
from torch import pi
def transform(Xs):
        return [torch.squeeze(x) for x in Xs]

#if __name__=='__main__':
from DataGenerator.HenonMapDataGen import HenonMapDataGen
from QuantumModels.QuantumSRNNs import QuantumSRNN
from QuantumModels.QuantumSRNNs import QuantumSystemFunction
from GradientFreeOptimizers.CostFunc import GradFreeMSELoss
import GradientFreeOptimizers.Helpers as hp

if __name__=='__main__':
    # Data Iter
    ## Parameters
    testSetRatio=0.2
    numStep=10
    batchSize=4
    currentPath=os.getcwd()
    savepath=os.path.join(currentPath,'data\HenonMap\Exp')
    filename='QExp1.csv'
    ## Generate Data
    '''
    hmap=HenonMapDataGen(savepath=savepath)
    hmap(1000)
    hmap.save_to_CSV(filename)
    '''
    ## Read the data
    hmap=HenonMapDataGen(savepath=savepath)
    hmap.read_from_CSV(filename)
    ## Get the Iter
    trainIter,testIter=hmap.get_data_iter(testSetRatio,numStep,batchSize,mask=0,shuffle=False)
    ## Print information
if __name__=='__main__':
    print(hmap)
    X,Y=next(iter(trainIter))
    print('Train Data Size:',len(trainIter))
    X,Y=next(iter(testIter))
    print('Test Data Size:',len(testIter))

# Model
## Parameters
inputSize=outputSize=1
qubits=4
activation=[0,2]
inputQubits=outputQubits=[i for i in range(qubits)]
interQPairs=[[i,j] for i in range(qubits) for j in range(i+1,qubits)]
inactive=['WIn','DeltaIn','J']
sysConstants={'Dissipation':None,'tau':pi,'steps':3,'numCpus':4}
measEffect=True
if __name__=='__main__':
    ## print parameters
    print('Input Qubits:',inputQubits)
    print('Output Qubits:',outputQubits)
    print('InterQPairs=',interQPairs)
#if __name__=='__main__':
## Get neccesary functions
srnnTestSup=QuantumSystemFunction()
#transform=lambda Xs:[torch.squeeze(x) for x in Xs]
init_rnn_state=srnnTestSup.get_init_state_fun(activation=activation)
get_params=srnnTestSup.get_get_params_fun(inputQubits=inputQubits,\
                                            outputQubits=outputQubits,\
                                            interQPairs=interQPairs,\
                                            inactive=inactive)
rnn=srnnTestSup.get_forward_fn_fun(measEffect=measEffect,\
                                        sysConstants=sysConstants)
predict_fun=srnnTestSup.get_predict_fun(outputTransoform=transform)

net=QuantumSRNN(inputSize,qubits,outputSize,get_params,init_rnn_state,rnn)
## Test prediction
if __name__=='__main__':
    state=net.begin_state(batchSize)
    Y,newState=net(X,state)
    print(Y.shape, len(newState), newState[0][0].shape)

    # Train the network
    ## Parameters
    num_epochs, lr = 10, 0.1
    step_epochs=1
    ## Loss function
    lossFunc=GradFreeMSELoss(net)
    ## Optimizer
    trainer = torch.optim.SGD(net.params, lr=lr)
    #scheduler=torch.optim.lr_scheduler.StepLR(trainer,step_size=100,gamma=0.1)
## Initial loss
if __name__=='__main__':
    l_epochs=[]
    train_l=QuantumSystemFunction.evaluate_accuracy(net,trainIter,lossFunc,False)
    test_l=QuantumSystemFunction.evaluate_accuracy(net,testIter,lossFunc,False)
    l_epochs.append([train_l,test_l])
    print('Initial Train Loss:',train_l)
    print('Initial Test Loss:',test_l)
    ## Training
    ## prediction
    predict = lambda prefix: predict_fun(prefix,net, numPreds=9)
    ## train and predict
    for epoch in range(num_epochs):
        timer=hp.Timer()
        trainLoss, speed = QuantumSystemFunction.train_epoch(
            net, trainIter, lossFunc, trainer, False)
        testLoss=QuantumSystemFunction.evaluate_accuracy(net, testIter, lossFunc, False)
        timeEpoch=timer.stop()
        if (epoch + 1) % step_epochs == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {trainLoss:.4f}, Test Loss: {testLoss:.4f},\
                 Time: {timeEpoch:.4f}s') 
        l_epochs.append([trainLoss,testLoss])
        #scheduler.step()
    testLoss=QuantumSystemFunction.evaluate_accuracy(net, testIter, lossFunc, False)
    print(f'TestLoss {testLoss:f}, {speed:f} point/s')

    # Prediction
    ## One-step prediction
    X,Y=next(iter(testIter))
    state=net.begin_state(batchSize)
    Y_hat,newState=net(X,state)
    Y=Y.transpose(0,1).reshape([-1,Y.shape[-1]])

    axes,fig=plt.subplots(1,1,figsize=(4,3))
    plt.title('One-Step Prediction')
    plt.plot(torch.linspace(0,Y.numel(),Y.numel()),torch.squeeze(Y),label='Y')
    plt.plot(torch.linspace(0,Y.numel(),Y.numel()),torch.squeeze(Y_hat).detach(),label=r'$\hat{Y}$')
    plt.legend()
    plt.show()

    ## Multi Step Prediction
    prefixSize=10
    totalSize=40
    testShift=int(len(hmap)*(1-testSetRatio))
    preX,preY=hmap.data_as_tensor
    preX,preY=torch.unsqueeze(preX[testShift:testShift+prefixSize],-1),torch.unsqueeze(preY[testShift:testShift+totalSize-1],-1)
    preY=[y for y in torch.cat((preX[:2],preY[1:]),dim=0)]
    preX=torch.unsqueeze(preX,-1)
    YHat=predict_fun(preX,net,numPreds=totalSize-prefixSize)

    axes,fig=plt.subplots(1,1,figsize=(4,3))
    plt.title('Multi-Step Prediction')
    fig.set_ylim(-2,2)
    plt.plot(torch.linspace(0,len(preY),len(preY)),preY,label='Y')
    plt.plot(torch.linspace(0,len(preY),len(preY)),YHat,label=r'$\hat{Y}$')
    plt.vlines([prefixSize-1],ymin=-2,ymax=2,linestyles='dashed',label='Prediction')
    plt.legend()
    plt.show()

    ## Parameters
    print(net.params)



