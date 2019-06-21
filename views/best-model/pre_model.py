from sklearn.externals import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score

bst = joblib.load('best_model_new_1_0.5_10_0.72_0.76.DataAnalysis')
data = pd.read_csv('test_data.csv')


X_test = data[['A','B','C']]
y = data['H']

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
dtest = xgb.DMatrix(X_test)

pred = bst.predict(dtest)
pred[pred<  0.75] = 0
pred[pred>= 0.75] = 1
# print('*' * 20, i, '*' * 20)
data['H'] = pred
print(pred.sum())
data.to_excel('test.xlsx')
# print(pred.sum())
# print(recall_score(y_test, pred))
# print(accuracy_score(y_test, pred))

# for i in np.linspace(0,1,21):
#     pred = bst.predict(dtest)
#     pred[pred< i] = 0
#     pred[pred>=i] = 1
#     print('*' * 20, i, '*' * 20)
#     print(pred.sum())
    # print(recall_score(y_test, pred))
    # print(accuracy_score(y_test, pred))