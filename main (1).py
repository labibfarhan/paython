'''
Note:
pip install bayespy
pip install pgmpy
'''
import pandas as pd
import numpy as np
# import bayespy as bp
import warnings 
warnings.filterwarnings('ignore')
heart_disease=pd.read_csv("data7_heart.csv")
# print(heart_disease)
print('Columns in the dataset')
for col in heart_disease.columns: 
    print(col) 
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
model=BayesianModel([('age','trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('exang',
'trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),('heartdisease','restecg'),
('heartdisease','thalach'), ('heartdisease','chol')])
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)
from pgmpy.inference import VariableElimination
HeartDisease_infer = VariableElimination(model)
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 1, 'sex' :0,'trestbps':150})
print(q)