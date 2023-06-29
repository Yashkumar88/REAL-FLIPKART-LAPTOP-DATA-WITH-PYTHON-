import streamlit as st;
import pandas as pd;
import numpy as np;
import pickle ;

df = pickle.load(open('df.pkl','rb'))


#--
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


x=df.drop(columns=['flipkart_offer'])
y=np.log(np.square(df['flipkart_offer']))

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.15,
                                               random_state=2)

step1 = ColumnTransformer([
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),['ProductName','CPU'])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe_1 = Pipeline(steps=[
    ('step1',step1),
    ('step2',step2)
])

pipe=pipe_1.fit(x_train,y_train)

#---
#-- title ---- 
st.title('LAPTOP-PRICE-PREDICTION')




# import 
pipe_1 = pickle.load(open('pipe_1.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

p = st.selectbox('ProductName',df['ProductName'].unique())

# total stars giving to laptops

stars =st.selectbox('STARS',df['stars'])

#  cpu

c =st.selectbox('CPU_NAME',df['CPU'].unique())

# TOTAL RAM
Ram= st.selectbox('Ram(in GB)',df['Ram'].unique())

# size
storage =st.selectbox('storage',df['storage'].unique())
# storage
size = st.selectbox('size (in cm)',df['size'].unique())

# actual_price 

actual_price =st.selectbox('ACTUAL_PRICE',df['actual_price'].unique())




# FLIPKART_OFFER

if st.button('Predicted_price'):
    if c =='acer':
        c= 0
    if c =='msi':
        c = 1
    if c == 'dell':
        c = 2
    if c == 'hp':
        c = 3
    if c == 'lenovo':
        c = 4
    if c == 'asus':
        c = 5
    if c == 'apple':
        c = 6
    if c =='other_brand':
        c = 7
    if c =='ASUS':
        c = 8
    if p == 'i9':
        p = 0
    if p == 'i7':
        p = 1
    if p == 'athlon':
        p = 2
    if p =='ryzen':
        p = 3
    elif p =='i5':
        p =4
    elif p == 'i3':
        p = 5
    elif p =='m1':
        p = 6
    elif p == 'celeron':
        p = 7

    elif p == 'm2':
        p = 8

    
    query=np.array([p,stars,c,Ram,size,storage,actual_price])
    
 #   query = np.array([ProductName,cpu,ram,size,storage,stars,actual_price])
    query = query.reshape(1,7)
    st.title((int(np.exp(pipe_1.predict(np.array(query))))))
    