#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import pandas as pd
import pickle
import tensorflow as tf
import sklearn.metrics as sk_metrics
import seaborn as sns


# In[2]:


def convert_labels(labels):
        """
            Convert label to numeric
        """
        label2indexes = dict((l, i) for i, l in enumerate(labels))
        index2labels = dict((i, l) for i, l in enumerate(labels))
            
        num_of_label = len(label2indexes)
        
        print('Label to Index: ', label2indexes)
        print('Index to Label: ', index2labels)
        print('number of labels: ', num_of_label)

        return label2indexes, index2labels

def onehot_encode1(x_raw, y_raw, char_dict, label_dict):
    """
    input: raw data, character dictionary, label dictionary
    output 27 entry numpy vector for each word and numeric y label
    """
    
    
    # Construct numpy vector with len(x_raw) rows and len(dictionary) columns
    x = np.zeros((len(x_raw), len(char_dict)))
    y_list = []
    
    for i, word in enumerate(x_raw):
        for letter in word:
            # find index of this letter from dict
            if letter not in char_dict:
                index = char_dict['UNK']
            else:
                index = char_dict[letter]
            x[i, index] += 1
    
        y_list.append(label_dict[y_raw[i]])
    
    # change list of y numeric labels into numpy array
    y = np.array(y_list)
    
    return x, y

def onehot_encode_onlyx(x_raw, char_dict):
    """
    input: raw data, character dictionary, label dictionary
    output 27 entry numpy vector for each word and numeric y label
    """
    
    
    # Construct numpy vector with len(x_raw) rows and len(dictionary) columns
    x = np.zeros((len(x_raw), len(char_dict)))
    
    for i, word in enumerate(x_raw):
        for letter in word:
            # find index of this letter from dict
            if letter not in char_dict:
                index = char_dict['UNK']
            else:
                index = char_dict[letter]
            x[i, index] += 1
    
    return x

def val_range_contains(value, range_str):
    """
    check if the given value is within the range 
    """
    if isinstance(range_str, str) and '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return start <= value <= end
    else:
        return value == range_str


# In[3]:


# import data20000 with clustering lables
pd_data20000_label_name = pd.read_csv("data20000_clusteringlabel_names.csv", header = 0)

#load char_dict
with open('char_dict.pickle', 'rb') as file:
    char_indices = pickle.load(file)

# load NN Model
model_5label = tf.keras.models.load_model('saved_model/model1')
model_5label.summary()


# ## Labeling

# In[4]:


# label index dictionary
label2index =  {'PTV': 0, 'CTV': 1, 'ICTV': 2, 'CTVn': 3, 'CTVp': 4}
index2label = {0: 'PTV', 1: 'CTV', 2: 'ICTV', 3: 'CTVn', 4: 'CTVp'}


# In[5]:


print(char_indices)


# ### CTV Label

# In[6]:


# CTV
CTV_clusters = [61, 110, 111, 113, 194, 282, 319, "349-496", 499, 500, 503, "522-525", 529, 
                531, 532, 624, "811-827", 857, 858, 984, 1000]
pd_CTV_label = pd_data20000_label_name[pd_data20000_label_name['label'].apply(lambda x: any(val_range_contains(x, val) for val in CTV_clusters))].copy()


# In[7]:


pd_CTV_label.shape


# In[8]:


# Predict
CTV_original = pd_CTV_label['original'].values
CTV_preprocess = pd_CTV_label['clean'].values
CTV_predict = onehot_encode_onlyx(CTV_preprocess, char_indices)
CTV_prediction = model_5label.predict(CTV_predict)
CTV_result = np.apply_along_axis(np.argmax, axis=1, arr=CTV_prediction)


# In[32]:


# change the resulting int categories back to string
CTV_result_label = []
for index in CTV_result:
    label = index2label[index]
    CTV_result_label.append(label)


# In[81]:


d = {'original': CTV_original, 'clean': CTV_preprocess,'index_category': CTV_result, 'label_category': CTV_result_label}
df_CTV_result = pd.DataFrame(data = d)


# In[37]:


df_CTV_result.to_csv('CTV_prediction.csv', index = False)


# In[5]:

# Manually check and revise the labeling then import back
# This is to show that with a preliminary model, how we can speed up the labeling process

re_CTV_result = pd.read_csv('/Users/maohuijun/Desktop/Rice/Yepes Lab/Prediction Result/re_CTV_prediction.csv')


# In[7]:


# Count accuracy
#.mean() returns the proportion of truth in boolean series
accuracy = (re_CTV_result['label_category'] == re_CTV_result['correct_label']).mean()


# ### A few more CTV

# In[302]:


CTV_clusters2 = [1024, 1097, 1098, 1131, 1132, 1137, 1156, 1163, 1169,  1207, '1226-1228', 
                1235, 1236, 1252, 1253, '1261 - 1319', 1339, 1347, 1348, 1349, 1411, 1471, 1565, 1570, 1575,
                '1597-1599', 1608, 1657, 1721, 1791, '1868-1872', 1901, 1902, 1926, 1942, '1953-1956', 
                '1966-1972']
pd_CTV_label2 = pd_data20000_label_name[pd_data20000_label_name['label'].apply(lambda x: any(val_range_contains(x, val) for val in CTV_clusters2))].copy()



# In[49]:


# Predict
CTV_original2 = pd_CTV_label2['original'].values
CTV_preprocess2 = pd_CTV_label2['clean'].values
CTV_predict2 = onehot_encode_onlyx(CTV_preprocess2, char_indices)
CTV_prediction2 = model_5label.predict(CTV_predict2)
CTV_result2 = np.apply_along_axis(np.argmax, axis=1, arr=CTV_prediction2)


# In[51]:


# change the resulting int categories back to string
CTV_result_label2 = []
for index in CTV_result2:
    label = index2label[index]
    CTV_result_label2.append(label)


# In[52]:


d = {'original': CTV_original2, 'clean': CTV_preprocess2,'index_category': CTV_result2, 'label_category': CTV_result_label2}
df_CTV_result2 = pd.DataFrame(data = d)


# In[53]:


df_CTV_result2.to_csv('CTV_prediction2.csv', index = False)


# In[6]:


re_CTV_result2 = pd.read_csv('/Users/maohuijun/Desktop/Rice/Yepes Lab/Prediction Result/re_CTV_prediction2.csv')


# In[104]:


# Count accuracy
accuracy = (re_CTV_result2['label_category'] == re_CTV_result2['correct_label']).mean()




# ### PTV

# In[305]:


PTV_clusters = [195, 242, 243, 285, 530, 863, 864, 1138, 1139, 1340, 1342, 1370, '1372-1387', 
                '1471-1534', 1602, '1975-1982', 2044, 2070, 2189, '2191-2195', 2418, 2419]
# select the PTV clusters only
# make a copy so that we are not modifying the original 
pd_PTV_label = pd_data20000_label_name[pd_data20000_label_name['label'].apply(lambda x: any(val_range_contains(x, val) for val in PTV_clusters))].copy()



# In[320]:


PTV_original = pd_PTV_label['original'].values
PTV_preprocess = pd_PTV_label['clean'].values
PTV_predict = onehot_encode_onlyx(PTV_preprocess, char_indices)
PTV_prediction = model_5label.predict(PTV_predict)
PTV_result = np.apply_along_axis(np.argmax, axis=1, arr=PTV_prediction)

PTV_result_label = []
for index in PTV_result:
    label = index2label[index]
    PTV_result_label.append(label)
    
d = {'original': PTV_original, 'clean': PTV_preprocess, 'index_category': PTV_result, 'label_category': PTV_result_label}
df_PTV_result = pd.DataFrame(data = d)



# In[52]:


df_PTV_result.to_csv('PTV_prediction.csv', index = False)


# In[7]:


re_PTV_result = pd.read_csv('/Users/maohuijun/Desktop/Rice/Yepes Lab/Prediction Result/re_PTV_prediction.csv')


# In[10]:


PTV_accuracy = (re_PTV_result['label_category'] == re_PTV_result['correct_label']).mean()



# In[8]:


# Total Accuracy 
pd_PTVCTV = pd.concat([re_PTV_result, re_CTV_result], axis = 0)


# In[20]:


Total_accuracy = (pd_PTVCTV['label_category'] == pd_PTVCTV['correct_label']).mean()


# ## Plot Distribution

# In[82]:


# Get all the clean data 
PTV_preprocess = pd_PTVCTV[pd_PTVCTV['correct_label'] == 'PTV']['clean'].values
CTV_preprocess = pd_PTVCTV[pd_PTVCTV['correct_label'] == 'CTV']['clean'].values
ICTV_preprocess = pd_PTVCTV[pd_PTVCTV['correct_label'] == 'ICTV']['clean'].values
CTVn_preprocess = pd_PTVCTV[pd_PTVCTV['correct_label'] == 'CTVn']['clean'].values
CTVp_preprocess = pd_PTVCTV[pd_PTVCTV['correct_label'] == 'CTVp']['clean'].values


# In[83]:


PTV_data = onehot_encode_onlyx(PTV_preprocess, char_indices)
CTV_data = onehot_encode_onlyx(CTV_preprocess, char_indices)
ICTV_data = onehot_encode_onlyx(ICTV_preprocess, char_indices)
CTVn_data = onehot_encode_onlyx(CTVn_preprocess, char_indices)
CTVp_data = onehot_encode_onlyx(CTVp_preprocess, char_indices)


# In[84]:


Probabilities_PTV = model_5label.predict(PTV_data)
Probabilities_CTV = model_5label.predict(CTV_data)
Probabilities_ICTV = model_5label.predict(ICTV_data)
Probabilities_CTVn = model_5label.predict(CTVn_data)
Probabilities_CTVp = model_5label.predict(CTVp_data )


# In[85]:


fig, axs = plt.subplots(5, 1, figsize=(12, 17))

axs[0].hist(Probabilities_PTV[:,0], bins=100, range=(0,1))
# axs[0].xlabel('Probability')
# axs[0].ylabel('Counts')
axs[0].set_title('PTV Probabilities Distribution')

axs[1].hist(Probabilities_CTV[:,1], bins=100, range=(0,1))
# axs[1].xlabel('Probability')
# axs[1].ylabel('Counts')
axs[1].set_title('CTV Probabilities Distribution')

axs[2].hist(Probabilities_ICTV[:,2], bins=100, range=(0,1))
# axs[2].xlabel('Probability')
# axs[2].ylabel('Counts')
axs[2].set_title('ICTV Probabilities Distribution')

axs[3].hist(Probabilities_CTVn[:,3], bins=100, range=(0,1))
# axs[3].xlabel('Probability')
# axs[3].ylabel('Counts')
axs[3].set_title('CTVn Probabilities Distribution')

axs[4].hist(Probabilities_CTVp[:,4], bins=100, range=(0,1))
# axs[4].xlabel('Probability')
# axs[4].ylabel('Counts')
axs[4].set_title('CTVp Probabilities Distribution')

fig.text(0.5, 0.02, 'Probability', ha='center', fontsize=12)
fig.text(0.04, 0.5, 'Counts', va='center', rotation='vertical', fontsize=12)


plt.show()



# ### ITV

# In[17]:


ITV_all = pd_data20000_label_name[pd_data20000_label_name['clean'].str.contains('itv', na = False)]


# In[18]:


ITV_all.to_csv('ITV_label.csv', index = False)


# In[9]:


ITV_label = pd.read_csv('/Users/maohuijun/Desktop/Rice/Yepes Lab/Prediction Result/ITV_label.csv')


# ### GTV

# In[10]:


GTV_clusters = ['33-36', 40, 89, 284, '526-528', 632, '660-756', 796, '829-853', 986, 988, 1164, 1166, 
                1169, 1226, 1228]
# make a copy so that we are not modifying the original 
pd_GTV_label = pd_data20000_label_name[pd_data20000_label_name['label'].apply(lambda x: any(val_range_contains(x, val) for val in GTV_clusters))].copy()
# add a column for y. Change all y to CTV first
pd_GTV_label.loc[:, 'correct_label'] = 'GTV'


# In[11]:


GTVp = ['gtvpt', 'gtvptzxl', 'gtvprimary', 'gtvp', 'gtvpbh', 'gtvpbhbh', 'gtvphu', 'gtvpp', 'gtvprimary',
       'gtvprimarybed', 'gtvprimarycopy', 'gtvprimarydef', 'gtvprimarypost', 'gtvprimaryprec', 'gtvprimaryprei',
       'gtvprimprecht', 'gtvprmary', 'gtvptt', 'gtvptxyj', 'gtvrnodepostc', 'gtvrnodeprech', 'gtvprimarynode', 
        'gtvprimnodes']
#IGTVn = ['igtvn', 'igtvnodes', 'igtvnodal', 'igtvnode']
#IGTVp = ['igtvp', 'igtvprimary']
#GTVcold = ['coldgtv']
GTVn = ['gtvnode', 'gtvn', 'gtvnt', 'gtvntjyc', 'gtvntt', 'gtvntxyj', 'gtvntzxl', 'gtvnbh', 'gtvnbhbh',
       'gtvnhu', 'gtvnmip', 'gtvndef', 'gtvngbg', 'gtvnleft', 'gtvnodal', 'gtvnodalnewscn', 'gtvnodalprechemo',
       'gtvnode', 'gtvnode?', 'gtvnodecopy', 'gtvnodem', 'gtvnodepostc', 'gtvnodeprech', 'gtvnodepet', 
       'gtvnode/', 'gtvnodebed', 'gtvnodebstres', 'gtvnodeece', 'gtvnodeexpand', 'gtvnodefnaneg', 'gtvnodegbg',
       'gtvnodeinterme', 'gtvnodepetleft', 'gtvnodepostcht', 'gtvnodepostin', 'gtvnodeprechem', 'gtvnodeprecht',
       'gtvnodepreindu', 'gtvnodepremfm', 'gtvnodepreop', 'gtvnoderightneck', 'gtvnodes', 'gtvnodescopy', 
       'gtvnodet', 'gtvnpre', 'gtvnprechemo', 'gtvnright', 'gtvprechtnodal', 'gtvprechemonod', 'nodalgtv',
       'nodegtv', 'nodesgtv']
#GTVboost = ['gtvboost', 'gtvboostgy', 'gtvnt', 'gtvntjyc', 'gtvntt', 'gtvntxyj', 'gtvntzxl', 'lnboostgtv']
#GTVpn = ['gtvprimarynode', 'gtvprimnodes']
#GTVring = ['gtvring']
IGTV = ['huigtv', 'igtv', 'igtvhu', 'igtvmm', 'igtvgy', 'igtvadapt', 'igtvadjhu', 'igtvall', 'igtvboost', 
       'igtvcopy', 'igtvesophagus', 'igtvhucorr', 'igtvhucorrct', 'igtvhucorrect', 'igtvhucrrect', 
       'igtvhuoriginal', 'igtvhuoverride', 'igtviitvmm', 'igtvinlung', 'igtvln', 'igtvlns', 'igtvlul',
       'igtvmd', 'igtvmediastinum', 'igtvnohu', 'igtvoverride', 'igtvrightscv', 'igtvtotal', 'igtvzl', 
       'originaligtv', 'igtvn', 'igtvnodes', 'igtvnodal', 'igtvnode', 'igtvp', 'igtvprimary']
CTVn = ['nodesctv']
ICTV = ['origictv', 'originalictv']

# relabel
pd_GTV_label.loc[pd_GTV_label['clean'].isin(GTVn), 'correct_label'] = 'GTVn'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(GTVp), 'correct_label'] = 'GTVp'
# pd_GTV_label.loc[pd_GTV_label['clean'].isin(IGTVn), 'y'] = 'IGTVn'
# pd_GTV_label.loc[pd_GTV_label['clean'].isin(IGTVp), 'y'] = 'IGTVp'
# pd_GTV_label.loc[pd_GTV_label['clean'].isin(GTVcold), 'y'] = 'GTVcold'
# pd_GTV_label.loc[pd_GTV_label['clean'].isin(GTVboost), 'y'] = 'GTVboost'
# pd_GTV_label.loc[pd_GTV_label['clean'].isin(GTVpn), 'y'] = 'GTVpn'
# pd_GTV_label.loc[pd_GTV_label['clean'].isin(GTVring), 'y'] = 'GTVring'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(IGTV), 'correct_label'] = 'IGTV'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(CTVn), 'correct_label'] = 'CTVn'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(ICTV), 'correct_label'] = 'ICTV'


# ## Encoding2

# In[12]:


# input all the labeled data
label1 = pd.read_csv('/Users/maohuijun/Desktop/Rice/Yepes Lab/Prediction Result/CTV_PTV_label1.csv')


# In[13]:


# label1: preliminary label for training, pd_PTVCTV: more labels, re_CTV_result2: more CTV
# ITV_label: ITV; pd_GTV_label: GTV
labelPTVCTV = pd.concat([label1, pd_PTVCTV, re_CTV_result2, ITV_label, pd_GTV_label], axis = 0)
preprocess_x = labelPTVCTV['clean'].values
preprocess_y = labelPTVCTV['correct_label'].values
original_x = labelPTVCTV['original'].values
labels = np.unique(preprocess_y)


# In[15]:


label2indexes_2, index2labels_2 = convert_labels(labels)


# In[16]:


def encode2(x_raw, y_raw, char_dict, label_dict):
    """
    input: x raw data, character dictionary
    output 30 entry numpy vector for each word
    """
    
    # Construct numpy vector with len(x_raw) rows and 30 columns
    x = np.zeros((len(x_raw), 30))
    y_list = []
    
    for i, word in enumerate(x_raw):
        position = 0
        for letter in word:
            # find index of this letter from dict
            if letter not in char_dict:
                index = char_dict['UNK']
            else:
                index = char_dict[letter]
            x[i, position] = index
            position += 1
    
        y_list.append(label_dict[y_raw[i]])
    # change list of y numeric labels into numpy array
    y = np.array(y_list)
    
    return x, y


# In[17]:


def encode2_onlyx(x_raw, char_dict):
    """
    input: x raw data, character dictionary
    output 30 entry numpy vector for each word
    """
    
    # Construct numpy vector with len(x_raw) rows and 30 columns
    x = np.zeros((len(x_raw), 30))
    
    for i, word in enumerate(x_raw):
        position = 0
        for letter in word:
            # find index of this letter from dict
            if letter not in char_dict:
                index = char_dict['UNK']
            else:
                index = char_dict[letter]
            x[i, position] = index
            position += 1
    
    return x



# In[19]:


x_encode2, y_encode2 = encode2(preprocess_x, preprocess_y, char_indices, label2indexes_2)




# In[23]:


# Define the input shape
tf.keras.backend.clear_session()
input_shape = [30]

# Create the model
model = tf.keras.models.Sequential()

# Add the input layer
model.add(tf.keras.layers.Input(shape=input_shape))

# Add other layers to the model
# model.add(tf.keras.layers.Dense(256, activation='sigmoid',
#                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(128, activation='sigmoid',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation='sigmoid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)))
#model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(32, activation='sigmoid',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
#model.add(tf.keras.layers.Dense(8, activation='sigmoid'))
#model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(x_encode2, y_encode2, test_size = 0.2, stratify=y_encode2, random_state=1)


# In[25]:


X_train_pre2, X_test_pre2, original_train_pre2, original_test_pre2 = train_test_split(preprocess_x, original_x, test_size = 0.2, stratify=y_encode2, random_state=1)


# In[27]:


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500)


# In[28]:


# Access the training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']


# In[29]:


# Create a range of epochs
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[30]:


train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']


# In[31]:


# Create a range of epochs
epochs = range(1, len(train_accuracy) + 1)
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[59]:


y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
cm = tf.math.confusion_matrix(y_test, y_pred)



# In[139]:


result_label = []
for index in y_pred:
    label = index2labels_2[index]
    result_label.append(label)


# In[140]:


correct_label = []
for index in y_test:
    label = index2labels_2[index]
    correct_label.append(label)


# In[141]:


d = {'original': original_test_pre2, 'clean': X_test_pre2,'index_category': y_pred, 'predicted_label': result_label, 'correct_label': correct_label}
df_encode2_result = pd.DataFrame(data = d)


# In[142]:


# label whether the prediction is right or wrong
df_encode2_result['prediction'] = 'correct'
df_encode2_result.loc[df_encode2_result['predicted_label'] != df_encode2_result['correct_label'], 'prediction'] = 'wrong'


# In[143]:


df_encode2_result.to_csv('encoding2_result.csv', index = False)


# In[266]:


plt.imshow(cm, cmap='viridis')
plt.colorbar()
plt.show()



# In[57]:


def show_confusion_matrix(test_labels, test_classes):
  # Compute confusion matrix and normalize
  plt.figure(figsize=(10,10))
  confusion = sk_metrics.confusion_matrix(test_labels, 
                                          test_classes)
  confusion_normalized = confusion / confusion.sum(axis=1)[:, np.newaxis]
  #confusion_normalized = confusion / confusion.sum(axis=1)
  axis_labels = range(10)
  ax = sns.heatmap(
      confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
      cmap='Blues', annot=True, fmt='.4f', square=True)
  plt.title("Confusion matrix")
  plt.ylabel("True label")
  plt.xlabel("Predicted label")




# In[157]:


# save model
get_ipython().system('mkdir -p saved_model')
model.save('saved_model/encode2_nooverfit_epoch500')




# In[132]:


# Find the data for each class 
CTV_index = np.where(y_test == 0)
CTV_data = X_test[CTV_index[0]]

CTVn_index = np.where(y_test == 1)
CTVn_data = X_test[CTVn_index[0]]

CTVp_index = np.where(y_test == 2)
CTVp_data = X_test[CTVp_index[0]]

GTV_index = np.where(y_test == 3)
GTV_data = X_test[GTV_index[0]]

GTVn_index = np.where(y_test == 4)
GTVn_data = X_test[GTVn_index[0]]

GTVp_index = np.where(y_test == 5)
GTVp_data = X_test[GTVp_index[0]]

ICTV_index = np.where(y_test == 6)
ICTV_data = X_test[ICTV_index[0]]

IGTV_index = np.where(y_test == 7)
IGTV_data = X_test[IGTV_index[0]]

ITV_index = np.where(y_test == 8)
ITV_data = X_test[ITV_index[0]]

PTV_index = np.where(y_test == 9)
PTV_data = X_test[PTV_index[0]]


# In[133]:


Probabilities_PTV = model.predict(PTV_data)
Probabilities_CTV = model.predict(CTV_data)
Probabilities_ICTV = model.predict(ICTV_data)
Probabilities_CTVn = model.predict(CTVn_data)
Probabilities_CTVp = model.predict(CTVp_data)
Probabilities_GTV = model.predict(GTV_data)
Probabilities_GTVn = model.predict(GTVn_data)
Probabilities_GTVp = model.predict(GTVp_data)
Probabilities_IGTV = model.predict(IGTV_data)
Probabilities_ITV = model.predict(ITV_data )


# In[142]:


fig, axs = plt.subplots(10, 1, figsize=(12, 35))

axs[0].hist(Probabilities_CTV[:,0], bins=100, range=(0,1))
# axs[0].xlabel('Probability')
# axs[0].ylabel('Counts')
axs[0].set_title('CTV Probabilities Distribution')

axs[1].hist(Probabilities_CTVn[:,1], bins=100, range=(0,1))
# axs[1].xlabel('Probability')
# axs[1].ylabel('Counts')
axs[1].set_title('CTVn Probabilities Distribution')

axs[2].hist(Probabilities_CTVp[:,2], bins=100, range=(0,1))
# axs[2].xlabel('Probability')
# axs[2].ylabel('Counts')
axs[2].set_title('CTVp Probabilities Distribution')

axs[3].hist(Probabilities_GTV[:,3], bins=100, range=(0,1))
# axs[3].xlabel('Probability')
# axs[3].ylabel('Counts')
axs[3].set_title('GTV Probabilities Distribution')

axs[4].hist(Probabilities_GTVn[:,4], bins=100, range=(0,1))
# axs[4].xlabel('Probability')
# axs[4].ylabel('Counts')
axs[4].set_title('GTVn Probabilities Distribution')

axs[5].hist(Probabilities_GTVp[:,5], bins=100, range=(0,1))
# axs[0].xlabel('Probability')
# axs[0].ylabel('Counts')
axs[5].set_title('GTVp Probabilities Distribution')

axs[6].hist(Probabilities_ICTV[:,6], bins=100, range=(0,1))
# axs[1].xlabel('Probability')
# axs[1].ylabel('Counts')
axs[6].set_title('ICTV Probabilities Distribution')

axs[7].hist(Probabilities_IGTV[:,7], bins=100, range=(0,1))
# axs[2].xlabel('Probability')
# axs[2].ylabel('Counts')
axs[7].set_title('IGTV Probabilities Distribution')

axs[8].hist(Probabilities_ITV[:,8], bins=100, range=(0,1))
# axs[3].xlabel('Probability')
# axs[3].ylabel('Counts')
axs[8].set_title('ITV Probabilities Distribution')

axs[9].hist(Probabilities_PTV[:,9], bins=100, range=(0,1))
# axs[4].xlabel('Probability')
# axs[4].ylabel('Counts')
axs[9].set_title('PTV Probabilities Distribution')

fig.text(0.5, 0.02, 'Probability', ha='center', fontsize=12)
fig.text(0.04, 0.5, 'Counts', va='center', rotation='vertical', fontsize=12)

#plt.savefig('prob_distri_test.png')

plt.show()


# ## Encoding1 model with more data 

# In[37]:


x_encode1, y_encode1 = onehot_encode1(preprocess_x, preprocess_y, char_indices, label2indexes_2)



# In[39]:


# Define the input shape
tf.keras.backend.clear_session()
input_shape = [27]

# Create the model
model1 = tf.keras.models.Sequential()

# Add the input layer
model1.add(tf.keras.layers.Input(shape=input_shape))

# Add other layers to the model
model1.add(tf.keras.layers.Dense(64, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model1.add(tf.keras.layers.Dropout(0.2))
model1.add(tf.keras.layers.Dense(32, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model1.add(tf.keras.layers.Dense(16, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
# model1.add(tf.keras.layers.Dropout(0.2))
model1.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model1.summary()


# In[40]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(x_encode1, y_encode1, test_size = 0.2, stratify=y_encode1, random_state=1)


# In[41]:


X_train_pre1, X_test_pre1, original_train_pre1, original_test_pre1 = train_test_split(preprocess_x, original_x, test_size = 0.2, stratify=y_encode1, random_state=1)



# In[44]:


history1 = model1.fit(X_train1, y_train1, validation_data=(X_test1, y_test1), epochs=500)


# In[45]:


# Access the training and validation loss
train_loss1 = history1.history['loss']
val_loss1 = history1.history['val_loss']


# In[46]:


# Create a range of epochs
epochs = range(1, len(train_loss1) + 1)
plt.plot(epochs, train_loss1, 'b', label='Training Loss')
plt.plot(epochs, val_loss1, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[47]:


train_accuracy1 = history1.history['accuracy']
val_accuracy1 = history1.history['val_accuracy']


# In[48]:


# Create a range of epochs
epochs = range(1, len(train_accuracy1) + 1)
plt.plot(epochs, train_accuracy1, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy1, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[49]:


string = np.array(['clinicaltumorvolume'])
encode_x = onehot_encode_onlyx(string, char_indices)
y_test = model1.predict(encode_x)
test_pred = np.argmax(y_test, axis=1)
test_label = index2labels_2[test_pred[0]]
test_label


# In[61]:


string = np.array(['cvt'])
encode_x = onehot_encode_onlyx(string, char_indices)
y_test = model1.predict(encode_x)
test_pred = np.argmax(y_test, axis=1)
test_label = index2labels_2[test_pred[0]]
test_label


# In[51]:


string = np.array(['primaryctv'])
encode_x = onehot_encode_onlyx(string, char_indices)
y_test = model1.predict(encode_x)
test_pred = np.argmax(y_test, axis=1)
test_label = index2labels_2[test_pred[0]]
test_label


# In[52]:


string = np.array(['nodectv'])
encode_x = onehot_encode_onlyx(string, char_indices)
y_test = model1.predict(encode_x)
test_pred = np.argmax(y_test, axis=1)
test_label = index2labels_2[test_pred[0]]
test_label


# In[53]:


string = np.array(['pvt'])
encode_x = onehot_encode_onlyx(string, char_indices)
y_test = model1.predict(encode_x)
test_pred = np.argmax(y_test, axis=1)
test_label = index2labels_2[test_pred[0]]
test_label


# In[54]:


y_pred1 = model1.predict(X_test1)
y_pred1 = np.argmax(y_pred1, axis=1)
cm1 = tf.math.confusion_matrix(y_test1, y_pred1)


# In[55]:


plt.imshow(cm1, cmap='viridis')
plt.colorbar()
plt.show()


# In[151]:


result_label = []
for index in y_pred1:
    label = index2labels_2[index]
    result_label.append(label)


# In[152]:


correct_label = []
for index in y_test1:
    label = index2labels_2[index]
    correct_label.append(label)


# In[153]:


d = {'original': original_test_pre1, 'clean': X_test_pre1,'index_category': y_pred1, 'predicted_label': result_label, 'correct_label': correct_label}
df_encode1_result = pd.DataFrame(data = d)


# In[154]:


# label whether the prediction is right or wrong
df_encode1_result['prediction'] = 'correct'
df_encode1_result.loc[df_encode1_result['predicted_label'] != df_encode1_result['correct_label'], 'prediction'] = 'wrong'


# In[155]:


df_encode1_result.to_csv('encoding1_result.csv', index = False)


# In[56]:


show_confusion_matrix(y_test1, y_pred1)


# In[158]:


# save model
get_ipython().system('mkdir -p saved_model')
model1.save('saved_model/encode1_nooverfit_epoch500')


# In[227]:


# Find the data for each class 
CTV_index = np.where(y_test1 == 0)
CTV_data = X_test1[CTV_index[0]]

CTVn_index = np.where(y_test1 == 1)
CTVn_data = X_test1[CTVn_index[0]]

CTVp_index = np.where(y_test1 == 2)
CTVp_data = X_test1[CTVp_index[0]]

GTV_index = np.where(y_test1 == 3)
GTV_data = X_test1[GTV_index[0]]

GTVn_index = np.where(y_test1 == 4)
GTVn_data = X_test1[GTVn_index[0]]

GTVp_index = np.where(y_test1 == 5)
GTVp_data = X_test1[GTVp_index[0]]

ICTV_index = np.where(y_test1 == 6)
ICTV_data = X_test1[ICTV_index[0]]

IGTV_index = np.where(y_test1 == 7)
IGTV_data = X_test1[IGTV_index[0]]

ITV_index = np.where(y_test1 == 8)
ITV_data = X_test1[ITV_index[0]]

PTV_index = np.where(y_test1 == 9)
PTV_data = X_test1[PTV_index[0]]


# In[229]:


Probabilities_PTV = model1.predict(PTV_data)
Probabilities_CTV = model1.predict(CTV_data)
Probabilities_ICTV = model1.predict(ICTV_data)
Probabilities_CTVn = model1.predict(CTVn_data)
Probabilities_CTVp = model1.predict(CTVp_data)
Probabilities_GTV = model1.predict(GTV_data)
Probabilities_GTVn = model1.predict(GTVn_data)
Probabilities_GTVp = model1.predict(GTVp_data)
Probabilities_IGTV = model1.predict(IGTV_data)
Probabilities_ITV = model1.predict(ITV_data )


# In[230]:


fig, axs = plt.subplots(10, 1, figsize=(12, 35))

axs[0].hist(Probabilities_CTV[:,0], bins=100, range=(0,1))
# axs[0].xlabel('Probability')
# axs[0].ylabel('Counts')
axs[0].set_title('CTV Probabilities Distribution')

axs[1].hist(Probabilities_CTVn[:,1], bins=100, range=(0,1))
# axs[1].xlabel('Probability')
# axs[1].ylabel('Counts')
axs[1].set_title('CTVn Probabilities Distribution')

axs[2].hist(Probabilities_CTVp[:,2], bins=100, range=(0,1))
# axs[2].xlabel('Probability')
# axs[2].ylabel('Counts')
axs[2].set_title('CTVp Probabilities Distribution')

axs[3].hist(Probabilities_GTV[:,3], bins=100, range=(0,1))
# axs[3].xlabel('Probability')
# axs[3].ylabel('Counts')
axs[3].set_title('GTV Probabilities Distribution')

axs[4].hist(Probabilities_GTVn[:,4], bins=100, range=(0,1))
# axs[4].xlabel('Probability')
# axs[4].ylabel('Counts')
axs[4].set_title('GTVn Probabilities Distribution')

axs[5].hist(Probabilities_GTVp[:,5], bins=100, range=(0,1))
# axs[0].xlabel('Probability')
# axs[0].ylabel('Counts')
axs[5].set_title('GTVp Probabilities Distribution')

axs[6].hist(Probabilities_ICTV[:,6], bins=100, range=(0,1))
# axs[1].xlabel('Probability')
# axs[1].ylabel('Counts')
axs[6].set_title('ICTV Probabilities Distribution')

axs[7].hist(Probabilities_IGTV[:,7], bins=100, range=(0,1))
# axs[2].xlabel('Probability')
# axs[2].ylabel('Counts')
axs[7].set_title('IGTV Probabilities Distribution')

axs[8].hist(Probabilities_ITV[:,8], bins=100, range=(0,1))
# axs[3].xlabel('Probability')
# axs[3].ylabel('Counts')
axs[8].set_title('ITV Probabilities Distribution')

axs[9].hist(Probabilities_PTV[:,9], bins=100, range=(0,1))
# axs[4].xlabel('Probability')
# axs[4].ylabel('Counts')
axs[9].set_title('PTV Probabilities Distribution')

fig.text(0.5, 0.02, 'Probability', ha='center', fontsize=12)
fig.text(0.04, 0.5, 'Counts', va='center', rotation='vertical', fontsize=12)

#plt.savefig('prob_distri_test.png')

plt.show()



