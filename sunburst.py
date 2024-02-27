# Databricks notebook source
import pandas as pd 
import plotly.express as px


folder_path =""


import chardet

# Open the file in binary mode and read a sample
with open(folder_path +'sales_data_sample.csv', 'rb') as f:
    result = chardet.detect(f.read(10000))  # Sample size of 10000 bytes

print(result['encoding'])
# Use the detected encoding to read the file
df = pd.read_csv(folder_path +'sales_data_sample.csv', encoding=result['encoding'])



#df = df.sort_values(by = 'SALES', ascending= False)
df['ordernumber'] = 'OrdNo.: ' + df['ORDERNUMBER'].astype('str')
df['orderedquantity'] ='Sold: '+ df['QUANTITYORDERED'].astype('str') + ' Units'
df['price each'] = 'Unit price: ' +df['PRICEEACH'].astype('str') + ' $'
df['deal size'] = 'Ord Size: ' + df['DEALSIZE']
df['sales'] = 'Sales: ' + df['SALES'].astype('str') + '$'


df1 = df.iloc[17:]



fig = px.sunburst(df1.head(10),path=['deal size', 'ordernumber', 'orderedquantity', 'price each'], values='SALES', color='SALES', height=1600, width=1600, color_continuous_scale='Portland', labels= 'SALES', hover_data=['SALES'], title = 'Multilayered sales information for each order')
display(fig)




