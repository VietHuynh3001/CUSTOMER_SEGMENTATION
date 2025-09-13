# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random 
from typing import Dict
import pickle




# HÀM
def readtxt(path):
    with open(path,encoding='utf-8',mode='r') as f:
        list_text=f.readlines()
    text=''.join(list_text)
    return text,list_text


def F(df):
    frequency=len(df)
    return frequency
    
def read_df(path):
    df=pd.read_csv(path)
    return df

def R(df):
    df_transactions=read_df('Combined_data.csv')
    df_transactions['Date']=pd.to_datetime(df_transactions['Date'],format='%Y-%m-%d')
    max_date=df_transactions['Date'].max().date()
    df['Date']=pd.to_datetime(df['Date'],format='%Y-%m-%d')
    recency=(max_date-df['Date'].max().date()).days
    return recency

def products_list(n):
    df_products=read_df('Products_with_Categories.csv')
    products=df_products['productName'].value_counts().index.to_list()
    random.seed(42)
    list_selected_products=random.sample(products,n)
    return list_selected_products


def query_product_pride(name):
    df_products=read_df('Products_with_Categories.csv')
    price = df_products[df_products['productName']==name]['price'].values[0]
    return price

def M(df):
    monetary=sum(query_product_pride(product)*quanity for row in df['Product'] for product,quanity in row.items())
    return monetary

def df_RFM(Recency,Frequency,Monetary):
    df_pred=pd.DataFrame({'Curency_scaled':[105],'Frenquency_scaled':[300],'Monetary_scaled':[400]})
    for column in df_pred.columns:
        df_pred[column]=np.log1p(df_pred[column])
    return df_pred

def customer_segmentation(df):
    with open('clustering_models.pkl','rb') as f:
        model=pickle.load(f)
    prediction=model.predict(df)[0]
    return prediction

def customer_group(label: int) -> str:
    mapping = {
        0: "Khách hàng trung thành",
        1: "Khách hàng ngủ quên",
        2: "Khách hàng phổ thông",
        3: "Khách hàng rời bỏ",
        4: "Khách hàng VIP cần giữ chân",
        5: "Khách hàng VIP hiện tại"
    }
    return mapping.get(label, "❓ Nhãn không hợp lệ")


def label_recency(r):
    if 263 <= r <= 349:
        return "Latest"
    elif 350 <= r <= 434:
        return "Middle"
    elif 435 <= r <= 520:
        return "Longest"
    else:
        return "Other"
        
def label_frequency(f):
    if 1 <= f <= 6:
        return "Low"
    elif 7 <= f <= 11:
        return "Medium"
    elif 12 <= f <= 16:
        return "High"
    else:
        return "Other"

def label_monetary(m):
    if 15 <= m <= 609:      
        return "Low"
    elif 609 < m <= 1204:   
        return "Medium"
    elif 1204 < m <= 1799:
        return "High"
    else:
        return "Other"


def assign_segment(r,f,m):
    if label_recency(r) == "Middle" and label_frequency(f) == "Low" and label_monetary(m) == "Low":
        return "Main Customers"
    elif label_recency(r) == "Longest" and label_frequency(f) == "High" and label_monetary(m) == "Low":
        return "Dormant Customers"
    elif label_recency(r) == "Middle" and label_frequency(f) == "High" and label_monetary(m) == "Low":
        return "Promising Customers"
    elif label_recency(r) == "Latest" and label_frequency(f) == "Low" and label_monetary(m) == "High":
        return "Potential New Customers"
    elif label_recency(r) == "Middle" and label_frequency(f) == "Low" and label_monetary(m) == "High":
        return "Potential Customers"
    else:
        return "Other"

