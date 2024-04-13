import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class HurricaneModel:

    ' Instance of a HurricaneModel has data and a model for time series forecasting'
    ' data must have the following columns: SID , date_time, and numeric Xfeatures and target'

    def __init__(self, model, data):
        self.model = model
        self.data = data

    # Adds lag as a feature, which is the target at the previous time stamp
    def addLaggedTarget(self, target, lag):
        self.data[target + '_lag'] = self.data[target].shift(lag)
        
        # Filling NA's with 0
        self.data.loc[0:lag-1, target + '_lag'] = 0

    
    def split_train_predict(self, ratio, X_feats, target, plot = False):
        
        # Splits data into training and testing based off of specified ratio for training data
        n = int(round(len(self.data) * ratio,1))
        train_df = self.data.head(n)
        test_df = self.data.tail(1-n)

        X_train = train_df[X_feats]
        y_train = train_df[target]
        X_test = test_df[X_feats]
        y_test = test_df[target]
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        # Accuracy Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        truth_arr, pred_arr = np.array(y_test), np.array(y_pred)
        mape = np.mean(np.abs((truth_arr - pred_arr) / truth_arr)) * 100

        
        print(str(pd.unique(self.data['SID'])) + " MSE: " + str(mse) + " MAE: "+ str(mae) + " MAPE: " + str(mape))

        # Plots predictions alongside original time series
        if plot == True:

            # Confidence intervals for data using a 2 step rolling average
            rolling_avg_dat = self.data[target].rolling(2).mean().combine_first(self.data[target]) 
            rolling_avg_test = pd.DataFrame(y_pred).rolling(2).mean().combine_first(pd.DataFrame(y_pred))

            se_dat =  2.576 * np.std(rolling_avg_dat) / math.sqrt(len(rolling_avg_dat))
            se_test =  2.576 * np.std(rolling_avg_test) / math.sqrt(len(rolling_avg_test))


            plt.figure(figsize = [10, 5])
            plt.title(pd.unique(self.data['SID']))

            plt.plot(self.data['date_time'], self.data[target])
            plt.plot(test_df['date_time'], y_pred)
            #plt.fill_between(self.data['date_time'], rolling_avg_dat - se_dat, rolling_avg_dat + se_dat, 
                            # alpha = 0.2, color = 'b')
            plt.fill_between(test_df['date_time'], (rolling_avg_test - se_test)[0], (rolling_avg_test + se_test)[0], 
                             alpha = 0.2, color = 'r')
            plt.axvline(x = test_df['date_time'].iloc[0], linestyle = '--', color = "green")
            plt.legend(['Actual ' + target, 'Predicted ' + target])
            plt.xticks(rotation = 45, size = 7)
            plt.ylabel(target)

   
    def plotTimeSeries(self, target):
        plt.plot(self.data['date_time'], self.data[target])
        plt.xticks(rotation = 90, size = 5)
        plt.ylabel(target)
        
        
    def plotACF(self, target):
        plot_acf(self.data[target])
        plt.show()


    def plotPACF(self, target):
        plot_pacf(self.data[target])
        plt.show()

    
    def plotDiff(self, target, order):
        diff_data = self.data.copy()

        # perform differencing
        for i in range(0, order):
            diff_data[target] = diff_data[target].diff().combine_first(diff_data[target])

        plt.figure(figsize = [7, 6])
        plt.plot(self.data['date_time'], self.data[target])
        plt.plot(diff_data['date_time'], diff_data[target], linestyle = '--', color = 'green')
        plt.legend(['Original', ' Order Difference: ' + str(order)])
        plt.xticks(rotation = 45, size = 7)
        plt.show()



    


        

        











    