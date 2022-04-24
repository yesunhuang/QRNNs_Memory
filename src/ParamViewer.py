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
import torch
import matplotlib.pyplot as plt

DRAW_LOSS=True
    #Save path:
if __name__=='__main__':
    currentPath=os.getcwd()
    netSavepath=os.path.join(currentPath,'TrainedNet','Exp')

if __name__=='__main__':
    filename='QExpFD.pt'


if  __name__=='__main__':
    netData=torch.load(os.path.join(netSavepath,filename))

    inputSize=netData['inputSize']
    outputSize=netData['outputSize']
    qubits=netData['qubits']
    
    inputQubits=netData['inputQubits']
    outputQubits=netData['outputQubits']
    activation=netData['activation']
    
    isDensity=netData['isDensity']
    interQPairs=netData['interQPairs']
    rescale=netData['rescale']
    inactive=netData['inactive']
    sysConstants=netData['sysConstants']
    measEffect=netData['measEffect']

    sysConstants['numCpus']=1

    print('params:\n','-'*40)
    for param in netData['NetParams']:
        print(param,'\n')
    print('constants:\n','-'*40)
    for constant in netData['NetConstants']:
        print(constant,'\n')

if DRAW_LOSS and __name__=='__main__':
    fig,axes=plt.subplots(1,1)
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    train_loss=[l[0] for l in netData['Loss']]
    test_loss=[l[1] for l in netData['Loss']]
    axes.plot(range(0,len(netData['Loss'])),train_loss,label='train loss')
    axes.plot(range(0,len(netData['Loss'])),test_loss,label='test loss')
    plt.legend()
    plt.show()