import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class HurricaneModel:

    ' Instance of a HurricaneModel has data and a model for time series forecasting'
    ' data must have the following columns: SID , date_time, and numeric Xfeatures and target'

    def __init__(self, model, data):
        self.model = model
        self.data = data

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
            ci = 1.645 * np.std(self.data[target])/np.mean(self.data[target])

            plt.figure(figsize = [10, 5])
            plt.title(pd.unique(self.data['SID']))

            plt.plot(self.data['date_time'], self.data[target])
            plt.plot(test_df['date_time'], y_pred)
            plt.axvline(x = test_df['date_time'].iloc[0], linestyle = '--', color = "green")
            plt.fill_between(self.data['date_time'], self.data[target] - ci, self.data[target] + ci, 
                             alpha = 0.1, color = 'b')
            
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
        plt.legend(['Original', str(order) + 'th Order Difference'])
        plt.xticks(rotation = 45, size = 7)
        plt.show()



    


        

        











    