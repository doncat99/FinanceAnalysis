import warnings
warnings.filterwarnings("ignore")
import faulthandler
faulthandler.enable()

import logging
from os import listdir
import time
from datetime import datetime
from collections import Counter

import pandas as pd
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from zvt import zvt_env
from zvt.contract.common import Region, Provider
from zvt.factors.candlestick_factor import CandleStickFactor, candlestick_patterns
# from zvt.contract.reader import DataReader
# from zvt.domain import Stock1dKdata, Stock
import zvt.stats as qs


logger = logging.getLogger(__name__)

def binary_class_classifier(current, future):
    return 1 if float(future) > float(current) else 0

def preprocess_df(df):
    except_col = ['target', 'Overnight_Return', 'ROC', 'ForceIndex', 'Momentum', 'Volatility']
    except_col += ['id', 'entity_id', 'timestamp']
    except_col += candlestick_patterns
    for col in df.columns:                                #
        if col not in except_col:
            df[col] = df[col].pct_change()
    df = df.replace([np.inf, -np.inf], np.nan) # to replace the infinite numbers by NAN
    df.dropna(inplace = True) # to drop NAN
    partial_df = df.iloc[:,:-1] # data without target column
    partial_np_scaled = preprocessing.scale(partial_df) #scaled data
    scaled_df = pd.DataFrame(partial_np_scaled, columns = df.columns[:-1], index =partial_df.index)
    scaled_df['target'] = df['target'].values
    return scaled_df

def Tech_Indicators(df):

    #Volatility #10
    df['Volatility']= df.close.pct_change().rolling(10).std()
    df['up'] = (df.close - df.open)
    
    #Create 10 days Moving Average
    df['SMA_10'] = df.close.rolling(window=10).mean()
    #Create Bollinger Bands()
    df['sd_using_Close'] = df.close.rolling(10).std()
    df['Upper_BB'] = df['SMA_10'] + (df['sd_using_Close']*2)
    df['Lower_BB'] = df['SMA_10'] - (df['sd_using_Close']*2)
    df = df.drop("sd_using_Close", axis =1)
    
    #Momentum  #10 #8  # 14 is even better  
    def momentum(Close):
        returns = np.log(Close)
        x = np.arange(len(returns))
        slope, _, rvalue, _, _ = linregress(x, returns)
        return ((1 + slope) ** 252) * (rvalue ** 2)
    df['Momentum'] = df.close.rolling(14).apply(momentum, raw=False)

    #Overnight returns
    df["Overnight_Return"] = df.open/df.close.shift(1)-1
    # Force Index #1
    df["ForceIndex"] = df.close.diff(1) * df.volume
    
    # Commodity Channel Index (CCI) #days =20 or 10(better) 5(even better)
    TP = (df.high + df.low + df.close) / 3 
    df['CCI'] = (TP - TP.rolling(5).mean()) / (0.015 * TP.rolling(5).std())

    # Ease Of Movement (EVM) #days =14
    dm = ((df.high + df.low)/2) - ((df.high.shift(1) + df.low.shift(1))/2)
    br = (df.volume / 100000000) / ((df.high - df.low))
    EVM = dm / br 
    df["EVM"] = EVM.rolling(14).mean()

    # Rate of Change (ROC) #5
    N = df.close.diff(7)
    D = df.close.shift(7)
    df['ROC'] = N/D

    df.dropna(inplace =True)

    return df

if __name__ == '__main__':
    now = time.time()
    pd.options.display.max_columns = 15
    pd.options.display.width = 10

    factor = CandleStickFactor(region=Region.US, 
                               codes=['FB', 'AMD'], 
                               start_timestamp='2015-01-01', 
                               kdata_overlap=0,
                               provider=Provider.Yahoo,
                               entity_provider=Provider.Yahoo)
    gb = factor.result_df.groupby('entity_id')
    dfs = {x : gb.get_group(x) for x in gb.groups}

    # df = dfs['AMD'][['open','close', 'volume','high', 'low']].copy()
    df = dfs['stock_NASDAQ_FB'].copy()
    df.set_index('timestamp', drop=True, inplace=True)

    df['future'] = df.open.shift(-1)
    df['target'] = list(map(binary_class_classifier, df.open, df.future))
    df.drop(['future', 'entity_id'], axis = 1, inplace =True)
    # df.dropna(inplace = True)

    print(df)
    df = preprocess_df(df)
    # print(df)
    splitting = int(0.70 * len(df))  #splitting ratio
    X_y_train = df[:splitting] #80% training set 
    X_y_test = df[splitting:]  #20% testing set
    X_y_train = X_y_train.sample(frac=1, random_state =123) # we shuffle the training set ONLY 

    X_train,y_train = X_y_train.iloc[:,:-1],X_y_train.iloc[:,-1]
    X_test,y_test = X_y_test.iloc[:,:-1], X_y_test.iloc[:,-1]

    #************************* Building the model **************************************************

    scale_pos_weight=Counter(y_train)[0]/Counter(y_train)[1]
    model = xgb.XGBRFClassifier(objective='binary:logistic',
                              scale_pos_weight=scale_pos_weight,
                              learning_rate=0.01,
                              n_estimators=5000,
                              max_depth=10,
                              min_child_weight=1,
                              gamma=0,
                              subsample=0.3,
                              colsample_bytree=0.3,
                              reg_alpha=0.014,
                              nthread=4,
                              seed=27) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_training = model.predict(X_train)
    acc_score = accuracy_score(y_test,y_pred)
    acc_score_training = accuracy_score(y_train,y_pred_training)

    print('Training Accuracy',acc_score_training)
    print('Testing Accuracy', acc_score)

    #********************** Features Importance *************************************************************

    xgb.plot_importance(model)
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.show()

    #************************ AUC ROC CURVE******************************************************************
    model_proba = model.predict_proba(X_test)
    model_proba = model_proba[:,1] #take only the probabilities that '1' is True
    model_auc = roc_auc_score(y_test,model_proba)
    print('Model AUC: ',model_auc)
    model_fpr, model_tpr,_ =  roc_curve(y_test,model_proba)

    random_proba =[0 for _ in range(len(y_test))] # we suppose that the random guesses are all 0
    random_auc = roc_auc_score(y_test,random_proba)
    random_fpr, random_tpr, _ = roc_curve(y_test, random_proba)

    plt.plot(random_fpr,random_tpr,linestyle ='--' ,label ='Random Prediction (AUROC =%0.2f)'% random_auc)
    plt.plot(model_fpr,model_tpr, label ='XGBoost (AUROC = %0.2f)'% model_auc)

    plt.title('ROC Plot')
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend()
    plt.show()





