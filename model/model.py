"""
create neural network model.
"""
import os
import re
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

batch_size = 1024
data_folder = "/data/traffic_violation"
df = pd.read_csv(os.path.join(data_folder, "Traffic_Violations.csv"),
                 usecols=['Description', 'Violation Type'])

df = df[df['Violation Type'].isin(['Warning', 'Citation'])]

for col in df.columns:
    df[col] = df[col].apply(lambda x: re.sub(r"\s+", " ", str(x).strip().upper()))

df = df.reset_index(drop=True)
max_len_text = int(df['Description'].map(lambda x: len(x)).quantile(0.99))
unique_characters = {x for y in df.Description for x in y}
character_dict = {y: x for x, y in enumerate(unique_characters)}
label_map = {"CITATION": 0, "WARNING": 1}

   
def train_test_val_split(df, p=[0.9, 0.05, 0.05]):
    choices = np.random.choice(["Train", "Validate", "Test"], size=len(df), replace=True, p=p)
    
    train_df = df[choices == 'Train'].reset_index(drop=True)
    test_df = df[choices == 'Test'].reset_index(drop=True)
    val_df = df[choices == 'Validate'].reset_index(drop=True)
    
    return train_df, test_df, val_df

train_df, test_df, val_df = train_test_val_split(df)


# def train_test_val_split(input_data, label, p=[0.85, 0.15, 0.0]):
#     choices = np.random.choice(["Train", "Validate", "Test"], size=len(input_data), replace=True, p=p)
#     train = (choices == "Train")
#     test = (choices == "Test")
#     val = (choices == "Validate")
    
#     Xtrain, ytrain = input_data[train, :, :], label[train, :]
#     Xtest, ytext = input_data[test, :, :], label[test, :]
#     Xval, yval = input_data[val, :, :], label[val, :]
    
#     return Xtrain, ytrain, Xtest, ytext, Xval, yval

#Xtrain, ytrain, Xtest, ytest, Xval, yval = train_test_val_split(input_arr, label)   

model = tf.keras.Sequential()
model.add(layers.Bidirectional(layers.LSTM(16, return_sequences=True), 
                               input_shape=(len(unique_characters), max_len_text)))
model.add(layers.Activation('tanh'))
model.add(layers.Bidirectional(layers.LSTM(8)))
model.add(layers.Activation('tanh'))
model.add(layers.Dense(8))
model.add(layers.Activation('softmax'))
model.summary()

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                filepath=checkpoint_path, 
                                                verbose=1, 
                                                save_best_only=False,
                                                save_weights_only=True,
                                                save_freq=5)

model.save_weights(checkpoint_path.format(epoch=0))
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer='sgd',
              metrics=['accuracy'],
              )      

def vectorize_txt(df_text):
    
    vectors = np.zeros((len(df_text), len(unique_characters), max_len_text))
    for i in range(len(df_text)):
        
        for k in range(min(max_len_text, len(df_text[i]) )):
            j = character_dict[df_text[i][k]]
            vectors[i, j, k] = 1
        
    return vectors
        

def gen(df=train_df, batch_size=batch_size):
    start = 0
    while start < len(df):
        df_tmp = df[start: start + batch_size].reset_index(drop=True)
        vector = vectorize_txt(df_tmp['Description'])
        label = df_tmp["Violation Type"].map(lambda x: label_map[x]).values
        yield vector, label
        start += batch_size
                
validation_generator = gen(val_df, batch_size=len(val_df))
Xval, yval = next(validation_generator)
train_gen = gen()

steps_per_epoch = int(np.ceil(len(train_df)/batch_size))

model.fit_generator(train_gen, validation_data=(Xval, yval), validation_steps=batch_size, callbacks=[cp_callback],
                    epochs=10, verbose=1, steps_per_epoch=steps_per_epoch, use_multiprocessing=False)

