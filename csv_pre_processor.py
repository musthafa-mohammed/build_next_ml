
import pandas as pd
import tensorflow as tf


postal_df = pd.read_csv("./saved_items/n.csv",sep=",")
pin_to_ditrict_map = postal_df[['PINCODE','DISTRICT']]




def pre_process():
    df = pd.read_csv("./saved_items/estimate_master.csv",sep=";")
    df = df[pd.notnull(df['pincode'])]
    df["district"] = get_district(df["pincode"])
    df.to_csv("test1.csv", sep=';', encoding='utf-8')


def get_district(pincode):
    district=[]
    for pin in pincode:
        a=str(pin_to_ditrict_map.loc[pin_to_ditrict_map['PINCODE']==int(pin),'DISTRICT'])
        t=a.split()
        #print(t[1])
        district.append(t[1])
    return district


pre_process()
