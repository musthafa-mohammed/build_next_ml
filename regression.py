import math

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

def regress(subcat_name):

    df = pd.read_csv("./saved_items/"+subcat_name+".csv",sep=";")
    df = df[(df[['area','SUM(ed.qty)']] != 0).all(axis=1)]
    df = df.reindex(np.random.permutation(df.index))






    def preprocess_features(df):
      processed_features = pd.DataFrame()
      processed_features['area'] = df['area']/df['area'].max()
      processed_features['district'] = df['district']
      processed_features.to_csv("./saved_items/x.csv", sep=';', encoding='utf-8')
      return processed_features


    def preprocess_targets(df):
      output_targets = pd.DataFrame()
      output_targets['SUM(ed.qty)'] = df['SUM(ed.qty)']/df['SUM(ed.qty)'].max()
      output_targets.to_csv("./saved_items/y.csv", sep=';', encoding='utf-8')
      return output_targets


    column_count = df.shape[0]
    column_count_80 = int(0.8*column_count)
    column_count_20 = column_count-column_count_80-1



    training_examples = preprocess_features(df.head(column_count_80))
    training_targets = preprocess_targets(df.head(column_count_80))

    new = pd.DataFrame()

    new['area'] = training_examples['area']
    new['qty'] = training_targets['SUM(ed.qty)']
    new.to_csv("./saved_items/new.csv", sep=';', encoding='utf-8')

    #assert not np.any(np.isnan(training_examples["area"]))
    #assert not np.any(np.isnan(training_targets["SUM(ed.qty)"]))

    validation_examples = preprocess_features(df.tail(column_count_20))
    validation_targets = preprocess_targets(df.tail(column_count_20))

    def construct_feature_columns(input_features):

        area_fe_cl = tf.feature_column.numeric_column("area")
        district_fe_cl = tf.feature_column.categorical_column_with_vocabulary_list(key="district",vocabulary_list=['Thiruvananthapuram','Kollam','Alappuzha','Pathanamthitta', 'Kottayam' ,'Idukki', 'Ernakulam' ,'Thrissur','Palakkad','Malappuram','Kozhikode','Wayanad','Kannur','Kasaragod'])
        return [area_fe_cl,district_fe_cl]



    def input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
        features = {key:np.array(value) for key,value in dict(features).items()}
        ds = Dataset.from_tensor_slices((features,targets))
        ds = ds.batch(batch_size).repeat(num_epochs)
        if shuffle:
          ds = ds.shuffle(10000)
        features, labels = ds.make_one_shot_iterator().get_next()
        return features, labels


    def train_model(learning_rate,steps,batch_size,training_examples,training_targets,validation_examples,validation_targets):
        periods = 10
        steps_per_period = steps / periods
        my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
        linear_regressor = tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_examples), optimizer=my_optimizer)
        training_input_fn = lambda:input_fn(training_examples,
                                          training_targets["SUM(ed.qty)"],
                                          batch_size=batch_size)
        predict_training_input_fn = lambda: input_fn(training_examples,
                                                      training_targets["SUM(ed.qty)"],
                                                      num_epochs=1,
                                                      shuffle=False)
        predict_validation_input_fn = lambda:input_fn(validation_examples,
                                                        validation_targets["SUM(ed.qty)"],
                                                        num_epochs=1,
                                                        shuffle=False)

        # Train the model, but do so inside a loop so that we can periodically assess
        # loss metrics.
        print("Training model...")
        print("RMSE (on training data):")
        training_rmse = []
        validation_rmse = []
        for period in range (0, periods):
            # Train the model, starting from the prior state.
            linear_regressor.train(input_fn=training_input_fn,steps=steps_per_period,)
            # Take a break and compute predictions.
            training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
            training_predictions = np.array([item['predictions'][0] for item in training_predictions])

            validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
            validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
            #print(validation_predictions)

            # Compute training and validation loss.
            training_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(training_predictions, training_targets))
            validation_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(validation_predictions, validation_targets))
            # Occasionally print the current loss.
            print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
            # Add the loss metrics from this period to our list.
            training_rmse.append(training_root_mean_squared_error)
            validation_rmse.append(validation_root_mean_squared_error)
        print("Model training finished.")


        # Output a graph of loss metrics over periods.
        plt.ylabel("RMSE")
        plt.xlabel("Periods")
        plt.title("Root Mean Squared Error vs. Periods")
        plt.tight_layout()
        plt.plot(training_rmse, label="training")
        plt.plot(validation_rmse, label="validation")
        plt.legend()
        #plt.show()


        return linear_regressor


    x=train_model(
    learning_rate=0.0001,
    steps=500,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

    # print("actual\n")
    # validation_examples = preprocess_features(df.tail(1))
    # validation_targets = preprocess_targets(df.tail(1))
    # print(validation_examples)
    # print("\n")
    # print(validation_targets)
    # print("\n")
    # predict_validation_input_fn = lambda:input_fn(validation_examples,
    #                                                 validation_targets["SUM(ed.qty)"],
    #                                                 num_epochs=1,
    #                                                 shuffle=False)
    # print("predicted")
    #
    # validation_predictions = x.predict(input_fn=predict_validation_input_fn)
    # validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    # print(validation_predictions)
    #xy = lambda:input_fn(validation_examples, validation_targets["SUM(ed.qty)"], num_epochs=1, shuffle=False)
    #z=x.predict(input_fn=xy)
    #print(z["predictions"][0])





regress("Cement")
