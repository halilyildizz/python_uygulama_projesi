#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Kütüphaneler #boş kayıt var mı ekle
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt


# In[2]:


# Tensorflow uyarı mesajlarının gizlenmesi..
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# In[4]:


# Veri kümesinin yüklenmesi..
missing_values = ["?"]
data = pd.read_csv("biopsy.csv",encoding='utf-8', na_values = missing_values)

print(data.head())


# In[5]:


data = data.dropna(axis="rows")
print(data.shape)


# In[6]:


print(data.head(10))


# In[7]:


print(data.describe().transpose())


# In[8]:


# Sınıf değişkeni ile diğerlerinin ayrılması ..
X = data.drop('Severity',axis=1)
y = data['Severity']


# In[9]:


# Önişlemler ..

# Kategorik niteliği sayısal değere dönüştürmek
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# Kategorik olan hedef değişkenin nümerik değere dönüştürülmesi
donusturuldu = LabelEncoder()
y= donusturuldu.fit_transform(y)

# Kategorik biçime dönüştürülmesi
y = np_utils.to_categorical(y)

# Verinin eğitim ve test biçiminde ayrılması
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1)


# In[10]:


# Verinin normalleştirilmesi
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(
    copy=True,
    with_mean=True,
    with_std=True)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[32]:


# Derin öğrenme mimarisi..
model = Sequential()
xx=50
model.add(Dense(
    xx,
    input_dim=4,            ### Öznitelik sayısı kadar olmalıdır..
    activation='relu'))

model.add(Dense(
    xx,
    activation='relu'))

model.add(Dense(
    xx,
    activation='relu'))

model.add(Dense(
    xx,
    activation='relu'))

model.add(Dense(
    xx,
    activation='relu'))

model.add(Dense(
    xx,
    activation='relu'))

model.add(Dense(
    xx,
    activation='relu'))

model.add(Dense(
    xx,
    activation='relu'))

model.add(Dense(
    xx,
    activation='relu'))

model.add(Dense(
    xx,
    activation='relu'))

model.add(Dense(
    xx,
    activation='relu'))

model.add(Dense(
    xx,
    activation='relu'))

model.add(Dense(
    xx,
    activation='relu'))

model.add(Dense(
    xx,
    activation='relu'))

model.add(Dense(
    2,                      ### Sınıf sayısı kadar olmalıdır..
    activation='softmax'))


# In[33]:


# Modelin derlenmesi
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# Modelin eğtitilmesi
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    verbose=2,
    batch_size=16,
    validation_split=0.3)


# In[34]:


# Eğitim performans grafikleri..
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Performansı')
plt.ylabel('doğruluk')
plt.xlabel('epoch')
plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')
plt.show()

# Hata grafiği ..
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model hataları')
plt.ylabel('hata')
plt.xlabel('epoch')
plt.legend(['Eğitim', 'Doğrulama'], loc='upper left')
plt.show()

# Test verisi ile performans ölçümü
scores = model.evaluate(X_test, y_test)
print("\nDoğruluk: ",(scores[1]))


# In[ ]:




