import math

import scipy as s

import matplotlib.pyplot as plt

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format



df = pd.read_csv("./saved_items/Cement.csv",sep=";")
df = df.reindex(np.random.permutation(df.index))



new=pd.DataFrame()


def preprocess_features(df):
  processed_features = pd.DataFrame()
  processed_features['area'] = df['area']/df['area'].max()
  new['area']=processed_features['area']
  processed_features['district'] = df['district']
  processed_features.to_csv("./saved_items/x.csv", sep=';', encoding='utf-8')
  return processed_features


def preprocess_targets(df):
  output_targets = pd.DataFrame()
  output_targets['SUM(ed.qty)'] = df['SUM(ed.qty)']/df['SUM(ed.qty)'].max()
  new['SUM(ed.qty)']=output_targets['SUM(ed.qty)']
  output_targets.to_csv("./saved_items/y.csv", sep=';', encoding='utf-8')
  return output_targets


column_count = df.shape[0]
column_count_80 = int(0.8*column_count)
column_count_20 = column_count-column_count_80-1

#new.plot(x='area', y='SUM(ed.qty)', style='o')


training_examples = preprocess_features(df.head(column_count_80))
training_targets = preprocess_targets(df.head(column_count_80))


correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["SUM(ed.qty)"]

print(correlation_dataframe.corr())

print(training_examples["area"].corr(training_targets["SUM(ed.qty)"]))

#plt.scatter(new['area'], new['SUM(ed.qty)'])
#plt.show()
