
import pandas as pd
import tensorflow as tf
from time import sleep


postal_df = pd.read_csv("./saved_items/n.csv",sep=",")
pin_to_ditrict_map = postal_df[['PINCODE','DISTRICT']]





def pre_process():
    print("Loading.......")
    df = pd.read_csv("/home/musthafa/Desktop/buildnext/work/build_next_ml/saved_items/new _data.csv",sep=",")
    df = df[pd.notnull(df['pincode'])]
    df = df[pd.notnull(df['area'])]
    df = df[pd.notnull(df['subcat_name'])]
    df = df[pd.notnull(df['SUM(ed.qty)'])]



    df["district"] = get_district(df["pincode"])
    #df.to_csv("test1.csv", sep=';', encoding='utf-8')
    df_new = df[['district','area','SUM(ed.qty)','subcat_name','pincode']]

    df_Cement = df_new.loc[df_new['subcat_name']=='Cement']
    df_Cement.to_csv("./saved_items/Cement.csv", sep=';', encoding='utf-8')



def get_district(pincode):
    district=[]
    for pin in pincode:
        a=str(pin_to_ditrict_map.loc[pin_to_ditrict_map['PINCODE']==int(pin),'DISTRICT'])
        t=a.split()
        #print(t[1])
        district.append(t[1])
    return district


pre_process()
