from config.path import  PROCESSED_DATA_DIR, MODELS_DIR
import time
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from DataPreprocesing import PreData,seq_prid
import pandas as pd
import numpy as np
import math
import os

class lstm_network(nn.Module):
    '''کلاس شبکه عصبی lstm '''
    def __init__(self,input_size=15,hidden_layer=50,output=1,num_lay=5):
        super(lstm_network,self).__init__()
        self.LSTM=nn.LSTM(input_size,hidden_layer,num_lay,batch_first=True,dropout=0.4)
        self.fc1=nn.Linear(hidden_layer,100)
        self.bn=nn.BatchNorm1d(100)
        self.activation=nn.LeakyReLU()
        self.dropout=nn.Dropout(0.2)
        self.fc2=nn.Linear(100,output)
    def forward(self,x:torch.tensor)->torch.Tensor:
        out,_=self.LSTM(x)
        x=out[:,-1,:]
        p=self.fc1(x)
        p=self.bn(p)
        p=self.activation(p)
        p=self.dropout(p)
        pred=self.fc2(p)
        return pred
    
def training():
    ''' اموزش مدل با داده هایی در ماژول DataPreprocesing 
    بهینه ساز RMSprop شده
    تابع هزینه :SMOOTHL1LOSS استفاده شده 
    مدل ارزیابی و تست شده
    '''
    start=time.time()
    torch.manual_seed(42)
    model=lstm_network()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    loss_func=nn.SmoothL1Loss()
    train_dataloeder,val_dataloeder,test_dataloeder=PreData(os.path.join(PROCESSED_DATA_DIR,"BTCUSD_andicator.csv"))
    val_loss_values=[]
    loss_val=[]
    best_val_loss=float('inf')
    patient=10
    counter=0
    epochs=[]


    for epoch in range(300):
        model.train()
        train_loss_epoch=0.0
        for batch_idx,(inputs,label) in enumerate(train_dataloeder):
            inputs,label=inputs.float() , label.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=loss_func(outputs,label)
            loss.backward()
            optimizer.step()
            train_loss_epoch+=loss.item()
        train_loss_epoch/=len(train_dataloeder)
        loss_val.append(train_loss_epoch)

        model.eval()
        val_loss=0
        with torch.no_grad():
            for inputs,label in val_dataloeder:
                outputs=model(inputs)
                loss=loss_func(outputs,label.unsqueeze(1))
                val_loss+=loss.item()
        val_loss/=len(val_dataloeder)
        val_loss_values.append(val_loss)
        epochs.append(epoch)
        print(f" epoch {epoch+1}/1000 , train loss: {train_loss_epoch:.4f} , val loss; {val_loss}")
        if val_loss < best_val_loss and epoch>=5:
            best_val_loss=val_loss
            counter=0
            torch.save(model.state_dict(),os.path.join(MODELS_DIR,"lstm.pth"))
            print(f"model saved with val loss: {best_val_loss}")
        else:
            counter+=1
            if counter >= patient:
                print(f"early stopping triggered after {epoch+1} epochs")
                break
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR,"lstm.pth"),weights_only=True))
    model.eval()
    test_loss_total=0
    mape_total=0
    n_sample=0
    rmse_total=0
    smape_total=0
    with torch.no_grad():
        for inputs,label in test_dataloeder:
            y_pred=model(inputs)
            test_loss=loss_func(y_pred,label.unsqueeze(1))
            test_loss_total+=test_loss.item()
            eps=1e-8
            mape=torch.abs((y_pred-label.unsqueeze(1)) / (label.unsqueeze(1) + eps))
            mape_total+=mape.sum().item()
            rmse=((y_pred - label.unsqueeze(1))**2).sum().item()
            rmse_total+=rmse
            smape = torch.abs(y_pred - label.unsqueeze(1)) / ((torch.abs(y_pred) + torch.abs(label.unsqueeze(1))) / 2 + eps)
            smape_total += smape.sum().item()
            n_sample+=label.size(0)
    avg_test_loss=test_loss_total/len(test_dataloeder)
    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'MAPE: {(mape_total / n_sample) * 100:.2f}%')
    rmse = math.sqrt(rmse_total / n_sample)
    print(f'RMSE: {rmse:.4f}')
    smape_avg = (smape_total / n_sample) * 100
    print(f'SMAPE: {smape_avg:.2f}%')
    
# training()


def predict():
    '''پیش بینی قیمت یک روز بعد با استفاده از داده های 110 روز قبل'''
    model=lstm_network()
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR,"lstm.pth"),weights_only=True))
    model.eval()
    data=pd.read_csv(os.path.join(PROCESSED_DATA_DIR,"BTCUSD_andicator.csv")).tail(110)
    print(data.tail(5)[["timestamp","close"]])
    print(f"min close: {data['close'].min()}")
    print(f"max close: {data['close'].max()}")
    print("input size:", data.drop(columns=["timestamp","close_time","close","ignore"],inplace=False).shape[1])
    sc_y=joblib.load(os.path.join(PROCESSED_DATA_DIR,"scaler_y.pkl"))
    sc_x=joblib.load(os.path.join(PROCESSED_DATA_DIR,"scaler_x.pkl"))
    

    data1=sc_x.transform(data.drop(columns=["timestamp","close_time","close","ignore"],inplace=False).values)
    print(F'min/max data:\n min: {data1.min()}\nmax: {data1.max()}')
    data2=seq_prid(data1,seq_len=110).unsqueeze(0)
    with torch.no_grad():
        prediction=model(data2)
    print(f"Raw output: {prediction}")
    org_pred=sc_y.inverse_transform(prediction.detach().numpy())
    return org_pred[-1].item()
print(predict())