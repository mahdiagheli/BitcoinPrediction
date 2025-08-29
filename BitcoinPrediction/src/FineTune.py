from config.path import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
import os
import requests
import pandas as pd
import torch
import torch.nn as nn
from lstm import lstm_network,predict
from DataPreprocesing import PreData,andicator,seq_prid
import math
def LastUpdate_read():
    """خواندن تاریخ اخرین اپدیت"""
    try:
        with open("LastUpdate.txt","r") as file:
            timestamp=int(file.read().strip())
        return timestamp
    except (FileNotFoundError,ValueError):
        return print("file not found,value error")
    
def SaveUpdate(timestamp):
    '''ذخیره تاریخ اپدیت به عنوان اخرین اپدیت'''
    with open("LastUpdate.txt","w") as file:
        file.write(str(int(timestamp)))


def NewData(lastupdate):
    '''دریافت داده های بروز از بایننس'''
    proxy={

    }
    urls=f"https://api.binance.com/api/v1/klines?symbol=BTCUSDT&interval=1h&startTime={lastupdate}"
    response=requests.get(url=urls,proxies=proxy,timeout=10)
    if response.status_code==200:
        print("The requests was successful")
    else:
        print(f"request faild! status code: {response.status_code}")
    data=response.json()
    if not data:
        print("No data recevied from api")
        return None
    df=pd.DataFrame(data,columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                     'quote_asset_volume', 'taker_buy_base', 'taker_buy_quote', 'number_of_trades'
        , 'ignore'
    ])
    return df
def UpdateDataset():
    '''افزودن داده های جدید به دیتاست قدیمی'''
    lastupdate=LastUpdate_read()
    new_data=NewData(lastupdate)
    if new_data is not None and not new_data.empty:
        dataset=pd.read_csv(os.path.join(RAW_DATA_DIR,"BTCUSD_combine.csv"))
        dataset=pd.concat([dataset,new_data],ignore_index=False).drop_duplicates(subset=["timestamp"])
        dataset.to_csv(os.path.join(PROCESSED_DATA_DIR,"BTCUSD_update.csv"),index=False)
        SaveUpdate(new_data["timestamp"].max())


def Fine_tune():
    '''  اپدیت وزن و بایاس مدل با استفاده از دیتاست جدید'''
    UpdateDataset()
    torch.manual_seed(42)
    model=lstm_network()
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR,"lstm.pth"),weights_only=True))
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    loss_func=nn.SmoothL1Loss()
    train_dataloeder,val_dataloeder,test_dataloeder=PreData(pd.read_csv(os.path.join(PROCESSED_DATA_DIR,"BTCUSD_andicator.csv")))
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
