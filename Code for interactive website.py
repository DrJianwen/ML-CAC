
from flask import Flask
from flask import Flask,jsonify,request
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEBUG = True

app = Flask(__name__)
CORS(app, supports_credentials=True)

file_path = os.path.dirname(os.path.abspath(__file__))



data=pd.read_excel(file_path+"/data_trainset.xlsx")
train_features = data.drop(columns=['CT_CAC_above400'])
train_target = data['CT_CAC_above400']

train_features['diabetes_duration'].fillna(train_features[train_features['diabetes_duration']!=0]['diabetes_duration'].median(),inplace=True)
train_features['hypertension_duration'].fillna(train_features[train_features['hypertension_duration']!=0]['hypertension_duration'].median(),inplace=True)
train_features.fillna(train_features.median(),inplace=True)



train_features_trans=train_features.copy()  
list_numerical = train_features_trans.drop(['gender_female',  'history_CVD', 'diabetes', 'hypertension'], axis=1).columns
print(list_numerical)
transfer = StandardScaler().fit(train_features_trans[list_numerical])   
train_features_trans[list_numerical] = transfer.transform(train_features_trans[list_numerical])   

svc_linear = SVC(kernel = "linear", C=0.19397, probability=True, class_weight='balanced', random_state=42)
svc_linear.fit(train_features_trans, train_target)  

@app.route('/', methods=['GET','POST'])
@app.route('/open/', methods=['GET','POST'])
@app.route('/open', methods=['GET','POST'])
def open():
    if request.method == 'POST':
        post_data = request.get_json()

        post_data = dict(post_data)
        print(post_data)
       
        key_words = post_data.keys()
        print(key_words)
        if "diabetes_duration" not in key_words:
            post_data['diabetes_duration'] = None
        if "hypertension_duration" not in key_words:
            post_data['hypertension_duration'] = None
        if "BNP" not in key_words:
            post_data['BNP'] = None
        if "ventricular_diameter" not in key_words:
            post_data['ventricular_diameter'] = None
        if "Red_blood_cell_count" not in key_words:
            post_data['Red_blood_cell_count'] = None
        
        if "diabetes" in key_words and post_data['diabetes'] == 0 :
            post_data['diabetes_duration'] =0
        if "hypertension" in key_words and post_data['hypertension'] == 0 :
            post_data['hypertension_duration'] =0

        test_features = pd.DataFrame(post_data,index=[0])
        test_features = test_features.rename(columns = {'age': 'age', 
            'Gender': 'gender_female',
            'diabetes': 'diabetes', 
            'hypertension': 'hypertension', 
            'Red_blood_cell_count': 'RBC_count', 
            'ventricular_diameter': 'LVESD', 
            'BNP': 'BNP', 
            'hypertension_duration': 'hypertension_duration', 
            'diabetes_duration': 'diabetes_duration', 
            'cerebrovascular':"history_CVD"})
        test_features =test_features[['age','gender_female','history_CVD','diabetes','diabetes_duration','hypertension','hypertension_duration','RBC_count','BNP','LVESD']]
        print(test_features)

        test_features['diabetes_duration'].fillna(train_features[train_features['diabetes_duration']!=0]['diabetes_duration'].median(),inplace=True)
        test_features['hypertension_duration'].fillna(train_features[train_features['hypertension_duration']!=0]['hypertension_duration'].median(),inplace=True)
        test_features.fillna(train_features.median(),inplace=True)
        print(test_features)

        test_features[list_numerical] = transfer.transform(test_features[list_numerical])

        test_proba = svc_linear.predict_proba(test_features)[:,1]
        test_proba = round(float(test_proba),3)
        print(test_proba)
        return jsonify(test_proba)
    return jsonify('ok')



if __name__ == '__main__':
    app.run(port=8000)

