from django.shortcuts import render
from django.http import HttpResponse,FileResponse
from django.views import View
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import xgboost as xgb
import time
import os
#from views.Pager import Pager

def predict_model(obj):
    # data = pd.read_excel("input.xlsx", index=None)
    data = pd.read_excel(obj)
    data["K"] = data['E'] + data["F"]
    data1 = data[["H", "A", "B", "C", "E", "F", "DATE", "TIME", "K"]]
    index = data1[(data1.K == 0)].index.tolist()
    data2 = data1.drop(index)
    data2['A'] = data2['A'].map(lambda x: float(x) * 100)
    data3 = data2.ix[data2['A'] >= 80]
    data4 = data3.ix[data3['A'] <= 1000]
    index = data4[(data4.H == 9)].index.tolist()
    data4.loc[index, 'H'] = 0
    data4["H"] = data4['H'].fillna(0)
    data4.loc[index, 'N'] = 1
    data4["N"].fillna(0, inplace=True)
    data4 = data4[['H', 'A', 'B', 'C', 'E', 'F', 'N', 'DATE', 'TIME']]
    data4['G'] = 0
    data4.to_excel('views/best-model/best_models/tem/test1190501.xlsx', index=None)

    bst = joblib.load('views/best-model/best_models/model/best_model_1-4.model')
    data = pd.read_excel('views/best-model/best_models/tem/test1190501.xlsx')
    shape = data.shape
    print(shape)
    count_one = int(shape[0] * 0.025)  # 1的总数
    X_test = data[['A', 'B', 'C']]
    dtest = xgb.DMatrix(X_test)
    pred = bst.predict(dtest)
    data['H'] = pred
    index_1 = data.sort_values(by='H', ascending=False).index
    index1_1 = index_1[0:count_one]
    data['H'] = 0
    data.loc[index1_1, 'H'] = 1

    bst1 = joblib.load('views/best-model/best_models/model/best_model_9-3.model')
    X_test1 = data[['A', 'B', 'C', 'H']]
    dtest = xgb.DMatrix(X_test1)
    pred1 = bst1.predict(dtest)
    data['N'] = pred1
    data['R'] = 0
    data.reset_index()

    grp = 1
    for i in range(data.shape[0] - 1):
        data.iloc[i, 9] = grp
        if data.iloc[i, 4] != data.iloc[i + 1, 4]:
            grp += 1
    for i in range(data.shape[0]):
        if data.iloc[i, 0] == 1:
            data.iloc[i, 6] = 0
    max_index = []
    count_index = 0
    for i in range(1, grp):
        index = data[(data.G == i)].index.tolist()
        data1 = data.iloc[index]
        value = np.array(data1.iloc[:, 6])
        max_index1 = np.argmax(value) + count_index
        count_index += len(index)
        max_index.append(max_index1)

    data2 = data[(data['H'] == 1)]
    value_count = sorted(list(set(data2['G'].values)))
    if value_count[-1] == data['G'].max():
        value_count.pop()

    for i in value_count:
        a = int(max_index[int(i)])
        data.iloc[a, 10] = 9
    data31 = data[["H", "E", "F", "N", "DATE", "TIME", "R", "A"]]

    def tran_to_date(x):
        if len(x) == 10:
            a = x[0:2]
            c = x[4:5]
            e = x[8:10]
            if x[2] == '0':
                b = x[3]
            else:
                b = x[2:4]
            if x[4] == '0':
                c = x[5]
            else:
                c = x[4:6]
            if x[6] == '0':
                d = x[7]
            else:
                d = x[6:8]
            time = "20" + a + "/" + b + "/" + c + " " + d + ':' + e
            return time

    data33 = data31

    def str1(x):
        x = str(int(x))
        if len(x) == 5:
            x = "0" + x
        return x

    data33['A'] = data33['A'].map(lambda x: round(float(x) / 100,3))
    data33['DATE'] = data33['DATE'].map(lambda x: str(x)[0:7])
    data33['TIME'] = data33['TIME'].map(lambda x: str1(x))
    data33['time'] = data33['DATE'] + data33['TIME']

    data4 = data33.sort_values(by="time", ascending=True)
    data4['time'] = data4['time'].map(lambda x: str(x)[1:-2])
    data4['time'] = data4['time'].map(lambda x: tran_to_date(x))
    data5 = data4[['time', 'H', 'R', 'E', 'F', 'A']]
    data5['R'] = data5['R'].fillna('0')
    data5['R'] = data5['R'].map(lambda x: float(x))
    data5['H'] = data5['H'].map(lambda x: float(x))
    data5['H'] = data5['R'] + data5['H']
    data5['R'] = data5['H']
    # data5.to_csv('result.csv', index=None)

    now_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    data_name = '{}-{}'.format(now_time,obj.name)
    data_path = 'load-data/' + data_name
    data5.to_excel('views/'+data_path)
    # print('保存成功')
    return data5,data_path,data_name

class LoadData(View):

    def get(self,request,page,data_name):
        if page:
            if data_name:
                data = pd.read_excel(os.path.join(os.path.dirname(__file__),'load-data',data_name))
                data = data[['time','H','R','E','F','A']]

                data_list = []
                for i in range(data.shape[0]):
                    # time, H, R, E, F, A
                    tmp = {'time': data.iloc[i, 0],
                           'H': data.iloc[i, 1],
                           'R': data.iloc[i, 2],
                           'E': data.iloc[i, 3],
                           'F': data.iloc[i, 4],
                           'A': data.iloc[i, 5], }
                    data_list.append(tmp)
                # obj = Pager(int(page), data.shape[0])
                # show_data = obj.get_data(data)
                # page_list = obj.show_page(data_name)

                return render(request, 'model.html', {'show_data': data_list})

        return render(request,'model.html')

    def post(self,request,page,data_name):
        obj = request.FILES.get('myfile')
        # DataAnalysis = request.POST.get('model_class')

        if obj:
            data,data_path,data_name = predict_model(obj)
            # data_path = os.path.join(os.path.dirname(__file__),data_path)
            data = data[['time','H','R','E','F','A']]

            data_list = []
            for i in range(data.shape[0]):
                # time, H, R, E, F, A
                tmp = {'time': data.iloc[i, 0],
                       'H': data.iloc[i, 1],
                       'R': data.iloc[i, 2],
                       'E': data.iloc[i, 3],
                       'F': data.iloc[i, 4],
                       'A': data.iloc[i, 5], }
                data_list.append(tmp)
            # page = 1
            # obj = Pager(page, data.shape[0])
            # show_data = obj.get_data(data)
            # page_list = obj.show_page(data_name)

            return render(request, 'model.html', {'show_data': data_list,'data_name':data_name})

        else:
            return render(request, 'model.html',{'msg':'请重新上传'})

def download(request,file_name):
    print(file_name)
    file = open(os.path.join(os.path.dirname(__file__),'load-data',file_name),'rb')
    response = FileResponse(file)
    print(file)
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename='+ file_name
    return response




