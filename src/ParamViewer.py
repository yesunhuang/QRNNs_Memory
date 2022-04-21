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

    #Save path:
if __name__=='__main__':
    currentPath=os.getcwd()
    netSavepath=os.path.join(currentPath,'TrainedNet','Exp')

if __name__=='__main__':
    filename='QExpF.pt'


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
