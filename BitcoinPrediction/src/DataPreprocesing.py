from config.path import RAW_DATA_DIR, PROCESSED_DATA_DIR, SCATTER_DIR
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import DataLoader,TensorDataset
import joblib
data_path=os.path.join(RAW_DATA_DIR,"BTCUSD_combine.csv")



def date():
    '''تبدیل timestamp به date'''
    df=pd.read_csv(data_path)
    df['Date']=pd.to_datetime(df['timestamp'],unit="ms")
    df['Date']=df['Date'].dt.strftime("%y-%m-%d")
    df=df.drop(columns=["timestamp"])
    save_path=os.path.join(PROCESSED_DATA_DIR,"BTCUSDT_combine.csv")
    return df.to_csv(save_path,index=False)
# date()

def out():
    '''شناسایی داده های پرت با IsolationForest'''
    data_path0=os.path.join(PROCESSED_DATA_DIR,"BTCUSDT_combine.csv")
    dataset=pd.read_csv(data_path0)
    os.makedirs("Scatter_out",exist_ok=True)
    for c in dataset.columns:
        dataset[c]=dataset[c].astype(float)
        x=dataset[c].values.reshape(1,-1)
        model=IsolationForest(contamination=0.01,random_state=42)
        y=model.fit_predict(x)
        plt.figure(figsize=(20,8))
        plt.scatter(range(len(x)),x[:,0],c="blue",label="normal",s=10)
        outlier=[i for i in range(len(y)) if y[i]==-1]
        plt.scatter(outlier,x[outlier],c="red",label="outlier",s=10)
        plt.xlabel("index")
        plt.title(f"outlier of{c}")
        plt.ylabel(f'{c}')
        plt.legend()
        plt.grid(True)
        img=os.path.join(SCATTER_DIR,"Scatter_out",f"{c}_outliers.png")
        plt.savefig(img,dpi=600)
        plt.close()
# out()

def andicator(dataset):
    '''بررسی داده های گمشده و افزودن اندیکاتور های فنی'''
    for i in dataset.columns:
        miss=dataset[i].isnull().sum()
        print(f"number of null in {i}: {miss}")
    dataset['SMA_14']=dataset['close'].rolling(window=14).mean()
    dataset['RSI_14']=ta.rsi(dataset['close'],length=14)
    macd=ta.macd(dataset['close'])
    dataset['MACD']=macd['MACDs_12_26_9']
    dataset['MACD-signal']=macd['MACDs_12_26_9']
    dataset["MACD-hist"]=macd['MACDs_12_26_9']
    bollinger=ta.bbands(dataset['close'],length=20)
    dataset['BB_upper']=bollinger['BBU_20_2.0']
    dataset['BB_lower']=bollinger['BBU_20_2.0']
    dataset['SMA_14']=dataset['SMA_14'].bfill()
    dataset['RSI_14']=dataset['RSI_14'].bfill()
    dataset['MACD']=dataset['MACD'].bfill()
    dataset['MACD-signal']=dataset['MACD-signal'].bfill()
    dataset['MACD-hist']=dataset['MACD-hist'].bfill()
    dataset['BB_upper']=dataset['BB_upper'].bfill()
    dataset['BB_lower']=dataset['BB_lower'].bfill()
    save_path=os.path.join(PROCESSED_DATA_DIR,"BTCUSDT_andicator.csv")
    dataset.to_csv(save_path,index=False)
# andicator()

def seq(x,y,seq_len):
    '''تبدیل داده به شکل مناسب برای ورودی lstm :
    (batch_size, seq_len, features)
    '''
    sequence=[]
    labels=[]
    for i in range(len(x)-seq_len):
        seq=x[i:i+seq_len]
        sequence.append(seq)
        label=y[i+seq_len-1]
        labels.append(label)
    return torch.stack(sequence),torch.tensor(labels,dtype=torch.float32)
def seq_prid(x,seq_len):
    '''تبدیل داده به شکل مناسب برای ورودی lstm :
    (batch_size, seq_len, features) برای پیش بینی'''
    if len(x)<seq_len:
        raise ValueError(f"data length {len(x)} is smaller than required seq_len {seq_len}")
    seq=x[-seq_len]
    return torch.tensor(seq, dtype=torch.float32).clone().detach().unsqueeze(0)

def PreData(dataset):
    '''تقسیم مجموعه داده به تست و اموزش و ارزیابی
       مقیاس بندی داده 
       ذخیره اسکالر 
       تقسیم به بچ 
    '''
    for colum in dataset.columns:
        print(f"min and max of {colum}:{dataset[colum].min()}----{dataset[colum].max()}")
    x=dataset.drop(columns=["timestamp","close_time","close","ignore"],inplace=False).iloc[1250:,:].values
    y=dataset.iloc[1250:,4].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42,shuffle=False)
    sc_x=RobustScaler()
    sc_y=RobustScaler()
    x_train0=torch.tensor(sc_x.fit_transform(x_train),dtype=torch.float32)
    x_test0=torch.tensor(sc_x.transform(x_test),dtype=torch.float32)
    y_train0=torch.tensor(sc_y.fit_transform(y_train.reshape(-1,1)).flatten(),dtype=torch.float32)
    y_test0=torch.tensor(sc_y.transform(y_test.reshape(-1,1)).flatten(),dtype=torch.float32)
    print(f" x_train rang:{x_train0.min()}, {x_train0.max()}")
    print(f"y_train rang:{y_train0.min()}, {y_train0.max()}")
    print(f"x_test rang:{x_test0.min()}, {x_test0.max()}")
    print(f" y_test rang:{y_test0.min()}, {y_test0.max()}")
    x_trainsq,y_trainsq=seq(x_train0,y_train0,seq_len=110)
    x_testsq,y_testsq=seq(x_test0,y_test0,seq_len=110)
    x_trainsq0,x_valsq,y_trainsq0,y_valsq=train_test_split(x_trainsq,y_trainsq,test_size=1/9,random_state=42,shuffle=False)
    train_dataloeder=DataLoader(TensorDataset(x_trainsq0,y_trainsq0),batch_size=28,shuffle=False)
    val_dataloeder=DataLoader(TensorDataset(x_valsq,y_valsq),batch_size=28,shuffle=False)
    test_dataloeder=DataLoader(TensorDataset(x_testsq,y_testsq),batch_size=28,shuffle=False)
    try:
        joblib.dump(sc_x,os.path.join(PROCESSED_DATA_DIR,"scaler_x.pkl"))
        print("saved sclaer_x to: scaler_x.pkl")
        joblib.dump(sc_y,os.path.join(PROCESSED_DATA_DIR,"scaler_y.pkl"))
        print("saved scaler_y to : scaler_y.pkl")
    except Exception as e:
        print(f"Error saving scaler; {e}")
        raise
    return train_dataloeder,val_dataloeder,test_dataloeder


























