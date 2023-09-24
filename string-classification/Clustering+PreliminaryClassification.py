#!/usr/bin/env python
# coding: utf-8

# ## Import library

# In[1]:


from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# In[2]:


pip install wordcloud


# In[2]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import pandas as pd
import pickle


# In[3]:


import tensorflow as tf



# ## Define Function

# In[4]:


def convert(list):
    """
    input list is a list of integer
    output list is one big integer
    the function transforms a list of integer into one big integer
    """
    # first convert the list
    str_list = [str(i) for i in list]
    # join the elements and change back to integer
    int_list = int("".join(str_list))
    
    return int_list


# In[5]:


def eva_dict(clustering, name_of_file):
    """
    input is a numpy array of clustering object 
    output is a dictionary that parallels original data with labels
    the function serves to evalutate the result of clustering
    """
    eva_dict = {}
    for index in range(len(name_of_file)):
        eva_dict[name_of_file[index]] = clustering.labels_[index]
    return eva_dict


# ## Import Data

# In[6]:


# Open left_example file in read mode
with open('Left_examples_all.txt', 'r') as file:
    # Read all the lines of the file into a list
    left_data = file.read().splitlines() 


# In[7]:


# Open SampleData_1500 
with open('SampleData_1500.txt', 'r') as file:
    # Read all the lines of the file into a list
    data1500 = file.read().splitlines() 


# In[8]:


# Open SampleData_20000
with open('SampleData_20000.txt', 'r') as file:
    # Read all the lines of the file into a list
    data20000 = file.read().splitlines() 


# ## Data Cleaning



# In[9]:


def cleaning4(list):
    
    """
    input a list of strings
    change into pd series, lowercase, unify left and right, get rid of special characters
    ouput the cleaned pandas dataframe (0, 'length')
    """
    
    # change into pandas series
    pd_data = pd.Series(list)
    
    # lowercase all the strings
    pd_data = pd_data.str.lower()
    
    # find the length of each string before cleaning
    pd_length = pd_data.str.len()
    
    # get rid of fs 
    fs = r'^fs'
    pd_data = pd_data.str.replace(fs, '')

    # get rid of numbers and brackets
    number_brac = r'[()\d]'
    pd_data = pd_data.str.replace(number_brac, '')

    # transform left and right 
    #left = r'(_l$)|(\sl$)|(\slt$)|(\sl\s)|(\slt\s)|(^l\s)|(^l_)|(^lt)|(_l\s)'
    left = r'(_l$)|(\sl$)|(\slt(?!o|e))|(\sl\s)|(\slt\s)|(^l\s)|(^l_)|(^lt(?!o|e))|(_l\s)|(-lt(?!o|e))|(_lf)|(-lf)|(_lt(?!o|e))|(_lft$)'
    
    pd_data = pd_data.str.replace(left, 'left')
    #pd_data = pd_data.str.match('left')

    # right 
    #right = r'(_r$)|(\sr$)|(\srt$)|(\sr\s)|(\srt\s)|(^r\s)|(^r_)|(_r\s)|(^rt(?!v))|(_rg)|(-rg)|(-rt$)'
    right = r'(_r$)|(\sr$)|(\srt$)|(\sr\s)|(\srt\s)|(^r\s)|(^r_)|(_r\s)|(^rt(?!v))|(_rg)|(-rg)|(-rt(?!v))|(_rt(?!v))'
    pd_data = pd_data.str.replace(right, 'right')
    
    # determine the special characters
    special_chars = r'[-_.\s()\d>%+]'
    pd_data = pd_data.str.replace(special_chars, '')
    
    # change the series into a dataframe
    df_data = pd_data.to_frame()
    
    # rename the column
    df_data = df_data.rename(columns={0: 'clean'})
    
    # add the length column
    df_data['length'] = pd_length
    
    return df_data


# In[10]:


# fourth data cleaning on data1500
# updated May 25, 2023
df_data1500_3 = cleaning4(data1500)


# In[11]:


df_data1500_3.head()



# In[10]:


df_data20000_2 = cleaning4(data20000)




# ## ASCII Transform

# In[11]:


# Define the calculation function for both methods
# Sum over character positions (ascii(c[i])*1000^(i-1)) where the last number occupies the largest digits
def reverse_ascii(textstr):
    return sum(ord(textstr[i]) * 1000**(i) for i in range(len(textstr)))


# In[12]:


def method1(pd_cleaned, column):
    """
    input the cleaned data in the form of pandas dataframe and column name
    output 2d numpy array reader for ML
    """
    # reverse every string in the pandas series
    pd_cleaned_reverse = pd_cleaned[column].apply(lambda x: x[::-1])
    # use the function from method 2 to change strings into ascii code
    pd_cleaned_ascii1 = pd_cleaned_reverse.apply(reverse_ascii)

    #concat ascii and length together
    pd_cleaned_ascii1_2d = pd.concat([pd_cleaned_ascii1, pd_cleaned['length']], axis = 1) 
    # change into numpy array
    np_cleaned_ascii1_2d = pd_cleaned_ascii1_2d.to_numpy()
    
    return np_cleaned_ascii1_2d


# In[13]:


def method2(pd_cleaned, column):
    """
    input the cleaned data in the form of pandas series
    output 2d numpy array reader for ML
    """
    # change into ascii code according to method 2 
    pd_cleaned_ascii2 = pd_cleaned[column].apply(reverse_ascii)

    #concat ascii and length together
    pd_cleaned_ascii2_2d = pd.concat([pd_cleaned_ascii2, pd_cleaned['length']], axis = 1) 
    
    # change into numpy array
    np_cleaned_ascii2_2d = pd_cleaned_ascii2_2d.to_numpy()
    
    return np_cleaned_ascii2_2d



# In[19]:


# Method 1 for fourth data cleaning on data1500
np_data1500_ascii1_2d_3 = method1(df_data1500_3, 'clean')
# Method 2
np_data1500_ascii2_2d_3 = method2(df_data1500_3, 'clean')


# In[14]:


# Method 1 for fourth data cleaning on data20000
np_data20000_ascii1_2d_2 = method1(df_data20000_2, 'clean')
# Method 2
np_data20000_ascii2_2d_2 = method2(df_data20000_2, 'clean')



# ## Machine Learning

# ### AP


# ### DBSCAN


# In[15]:


def dbscan_labels_outliers(np_array):
    """
    input numpy array 
    print out the number of nonoutliers and labels
    return the clustering object
    """
    # execute DBSCAN 
    clustering_DBSCAN = DBSCAN(eps=3, min_samples=2).fit(np_array)
    
    # find the number of nonoutliers 
    nonoutliers = len(clustering_DBSCAN.core_sample_indices_)
    
    # find the number of labels
    labels = len(np.unique(clustering_DBSCAN.labels_))

    print("There are", nonoutliers, "nonoutliers")
    print("There are", labels, "labels")
    
    return clustering_DBSCAN




# In[16]:


# Updated May 26, 2023 for fourth data cleaning method 
print('result for method 1')
clustering_DBSCAN_data1500_ascii1_3 = dbscan_labels_outliers(np_data1500_ascii1_2d_3)
print('result for method 2')
clustering_DBSCAN_data1500_ascii2_3 = dbscan_labels_outliers(np_data1500_ascii2_2d_3)


# In[16]:


# Updated May 30, 2023 for fourth data cleaning method on data20000
print('result for method 1')
clustering_DBSCAN_data20000_ascii1_2 = dbscan_labels_outliers(np_data20000_ascii1_2d_2)
print('result for method 2')
clustering_DBSCAN_data20000_ascii2_2 = dbscan_labels_outliers(np_data20000_ascii2_2d_2)



# In[17]:


# Def evaluation csv function
def eva_pd(clustering_object, raw_data):
    """
    input clustering object, and raw data in the form of list
    output pandas dataframe of (label, names)
    """
    # Change all labels into pandas series
    pd_clustering_object_labels = pd.Series(clustering_object.labels_)
    
    # Change raw data into pandas
    pd_raw_data = pd.Series(raw_data)
    
    # Concat the two
    pd_conc = pd.concat([pd_raw_data, pd_clustering_object_labels], axis = 1)
    
    # group by labels and put all the names in the same columns
    groups = pd_conc.groupby(1)[0].apply(lambda x: ', '.join(x)).reset_index()
    
    #rename the column name 
    groups.columns = ['label', 'names']
    
    return groups



# In[23]:


# ---Evaluation, data1500, fourth data cleaning ---

# Method 1
groups_ascii1_3 = eva_pd(clustering_DBSCAN_data1500_ascii1_3, data1500)
# Method 2
groups_ascii2_3 = eva_pd(clustering_DBSCAN_data1500_ascii2_3, data1500)


# In[96]:


# OPTIONAL save as a csv file

# Method 1
groups_ascii1_3.to_csv('ascii1_result_3.csv', index = False)
# Method 2
groups_ascii2_3.to_csv('ascii2_result_3.csv', index = False)


# In[18]:


# ---Evaluation, data20000, fourth data cleaning ---

# Method 1
groups_ascii1_data20000_2 = eva_pd(clustering_DBSCAN_data20000_ascii1_2, data20000)
# Method 2
groups_ascii2_data20000_2 = eva_pd(clustering_DBSCAN_data20000_ascii2_2, data20000)


# In[53]:


# OPTIONAL save as a csv file

# Method 1
groups_ascii1_data20000_2.to_csv('ascii1_data20000_result_2.csv', index = False)
# Method 2
groups_ascii2_data20000_2.to_csv('ascii2_data20000_result_2.csv', index = False)


# ## Automatic Label

# In[40]:


# Define function for automatic label
def auto_label(eva_pd_groups):
    """
    input evaluation 2D pandas dataframe(label, names)
    output groups(string_label, names, count) and a frequency table for wordcloud
    """
    # make a copy of the input dataframe
    groups = eva_pd_groups.copy()
    
    # calculate the number of terms in each label
    groups['count'] = groups['names'].apply(lambda x: x.count(',') + 1)
    
    # find the first element in names column, this regex will result in two columns
    fre_labels = groups['names'].str.extract('(^[0-9]+)|(^[A-Za-z]+)')
    # combine the two columns into a new one
    fre_labels['label'] = fre_labels[0].fillna(fre_labels[1])
    # drop the two old columns
    fre_labels = fre_labels.drop([0, 1], axis=1)
    # lowercase all the labels
    fre_labels['label'] = fre_labels['label'].str.lower()

    # replace the numeric label in input dataframe with auto extracted label
    groups['label'] = fre_labels['label']
    
    
    # change the series into dataframe
    #df_fre_labels = fre_labels.to_frame()
    # add a new column for the count of terms for each cluster
    fre_labels['frequency'] = groups['count']
    # drop the outliers row
    fre_labels = fre_labels.drop(0)
    # group by labels, there are 117 labels
    fre_labels = fre_labels.groupby(['label']).sum()

    # change the df of frequency column into a dict 
    fre_labels_dict = fre_labels['frequency'].to_dict()
    
    return groups, fre_labels_dict


# In[41]:


# Auto Label for second data cleaning
df_ascii1, ascii1_dict_wc = auto_label(groups_ascii1)
df_ascii2, ascii2_dict_wc = auto_label(groups_ascii2)


# In[42]:


# Auto Label for third data cleaning
df_ascii1_2, ascii1_dict_wc_2 = auto_label(groups_ascii1_2)
df_ascii2_2, ascii2_dict_wc_2 = auto_label(groups_ascii2_2)


# In[43]:


# Auto Label for data20000
# Method 1
data20000_ascii1_2, data20000_ascii1_dict_wc = auto_label(groups_ascii1_data20000)
# Method 2
data20000_ascii2_2, data20000_ascii2_dict_wc = auto_label(groups_ascii2_data20000)


# In[63]:


data20000_ascii1_2.head()



# ## WordCloud

# In[44]:


# define function for wordcloud visualization based on frequency table
def wc_fre_vis(dict):
    """
    input frequency table
    output wordcloud visualization and wc object
    """
    
    wc = WordCloud(font_path="./timr45w.ttf", background_color="black", width = 1200, height = 1000, max_words=2000, min_font_size = 10, collocations=False, contour_width=3, colormap = 'Pastel1').generate_from_frequencies(dict)
    plt.axis("off")
    # so that the size shown matches our request 
    plt.figure(figsize=(12,10))
    plt.tight_layout()
    plt.imshow(wc)
    
    return wc


# In[93]:


# ---fourth data cleaning visulization, data1500---

wc_ascii1_3 = wc_fre_vis(ascii1_dict_wc_3)
wc_ascii2_3 = wc_fre_vis(ascii2_dict_wc_3)


# In[94]:


# ---fourth data cleaning visulization, data20000---

wc_ascii1_data20000_3 = wc_fre_vis(data20000_ascii1_dict_wc_3)
wc_ascii2_data20000_3 = wc_fre_vis(data20000_ascii2_dict_wc_3)




# In[96]:

# OPTIONAL save image as png file locally, cleaning4

# data1500
wc_ascii1_3.to_file('DBSCAN_ascii1_data1500_3_wc.png')
wc_ascii2_3.to_file('DBSCAN_ascii2_data1500_3_wc.png')

# data20000
wc_ascii1_data20000_3.to_file('DBSCAN_ascii1_data20000_3_wc.png')
wc_ascii2_data20000_3.to_file('DBSCAN_ascii2_data20000_3_wc.png')




# ## Supervised learning
# 

# ### Prepare data for labeling

# In[32]:


type(clustering_DBSCAN_data20000_ascii1_2)


# In[19]:


# Prepare for label for supervised learning through pandas
# Create a df of form (clustering label, original string, input ascii data) 

# Change raw data into pandas
pd_raw_data20000 = pd.DataFrame(data20000, columns = ['original'])

# change ascii clustering input data into 1D pandas dataframe
pd_ascii_length_data20000 = pd.DataFrame(np_data20000_ascii1_2d_2, columns = ['ascii', 'length'])
pd_ascii_length_data20000['ascii_length'] = pd_ascii_length_data20000.apply(lambda row: [row['ascii'], row['length']], axis = 1)
pd_ascii_length_data20000 = pd_ascii_length_data20000.drop(columns = ['ascii', 'length'])

# Concatenate raw data and ascii data
pd_data20000_ori_ascii = pd.concat([pd_raw_data20000, pd_ascii_length_data20000], axis = 1)

# change clustering labels into pandas dataframe
pd_clustering_DBSCAN_data20000_ascii1_2 = pd.DataFrame(clustering_DBSCAN_data20000_ascii1_2.labels_, columns = ['label'])

# concatenate labels with the names
pd_data20000_label_name = pd.concat([pd_clustering_DBSCAN_data20000_ascii1_2, pd_data20000_ori_ascii], axis = 1)


# In[20]:


# concatentate the cleaned name with the original name and labels 
pd_data20000_label_name = pd.concat([pd_data20000_label_name, df_data20000_2], axis = 1)


# In[21]:


pd_data20000_label_name.to_csv('data20000_clusteringlabel_names.csv', index = False)


# ### Preliminary Label CTV, PTV

# In[22]:


# CTV Cluster: 13, 14, 38, 29
# put all CTV Cluster into a new dataframe
pd_CTV_data20000 = pd_data20000_label_name[pd_data20000_label_name['label'] == 13]
pd_CTV_data20000 = pd.concat([pd_data20000_label_name[pd_data20000_label_name['label'] == 14], pd_CTV_data20000], axis = 0)
pd_CTV_data20000 = pd.concat([pd_data20000_label_name[pd_data20000_label_name['label'] == 38], pd_CTV_data20000], axis = 0)
pd_CTV_data20000 = pd.concat([pd_data20000_label_name[pd_data20000_label_name['label'] == 39], pd_CTV_data20000], axis = 0)

# add a column for y. Change all y to CTV first
pd_CTV_data20000['y'] = 'CTV'

# find all ICTV, CTVn, CTVp, CTVt
ICTV = ['2iCTV_T50 zxl', 'iCTV_T50 zxl', '2iCTV_T0', '2iCTV_T30', 'iCTV_T0', 'ICTVT0', 'iCTV_T30', 'ICTV_T50']
CTVn = ['1CTV_6930_n', 'CTV 1 70n', 'CTV 2 63n', 'CTV 3 57n', 'CTV_50_n', 'CTV_54_n', 'CTV69_N', 'CTVn', 'CTV_N1',
       'CTVn_4140']
CTVp = ['1CTV_6930_p', 'CTV_58_p', 'CTV63_P', 'CTV69_P', 'CTVp', 'CTV_P', 'CTV_P1', 'CTVp1_6750', 'CTVP+5',
       'CTVp_50']
# #CTVt = ['CTV57_T', 'CTV59_T', 'CTV60T', 'CTV 63_T', 'CTV63_T', 'CTV_T0', 'CTV_T00', 'ctv T00_6.22', 'CTV_T0_3.1',
#        'CTV_T30', 'ctv T30_6.22', 'CTV_T0_3.1', 'ctv T30_6.22', 'CTV_T50', 'CTV_T50_1', 'CTV_T50_2.11',
#        'CTV_T50_2.17', 'CTV_T50_3.1', 'CTV_T51', 'CTV_T52', 'CTV_T53', 'CTV_T54', 'CTVt_6750', 'CTV_T80',
#        'ctv T80_6.22']

# Relabel the dataframe
pd_CTV_data20000.loc[pd_CTV_data20000['original'].isin(ICTV), 'y'] = 'ICTV'
pd_CTV_data20000.loc[pd_CTV_data20000['original'].isin(CTVn), 'y'] = 'CTVn'
pd_CTV_data20000.loc[pd_CTV_data20000['original'].isin(CTVp), 'y'] = 'CTVp'
#pd_CTV_data20000.loc[pd_CTV_data20000['original'].isin(CTVt), 'y'] = 'CTVt'



# In[24]:


# PTV Cluster: 18, 19, 20, 21, 22, 52, 87, 113, 136 - 139
# put all PTV Luster into a new dataframe
PTV_clusters = [18, 19, 20, 21, 22, 52, 87, 113, 136, 137, 138, 139]
pd_PTV_data20000 = pd_data20000_label_name[pd_data20000_label_name['label'].isin(PTV_clusters)]

# add a column for y. Change all y to PTV first
pd_PTV_data20000.loc[:,'y'] = 'PTV'

# Find all 
# PTVring = ['1PTV12ring', '1PTV1ring', '1PTVall1cmring', '1PTVall2cmring', '1PTVall3cmring', '1PTVall4cmring', 
#           '1PTVall5cmring', 'AP_PTV12ring', 'AP_PTVall1cmring', 'AP_PTVall2cmring', 'AP_PTVall3cmring', 
#           'AP_PTVall4cmring', 'AP_PTVall5mmring', 'PTVring', 'PTV_ring', 'PTVRing', 'PTV_Ring', 'PTV_RING']
# PTVplan = ['1PTV2plan', 'AP_PTV2plan', 'PTV-plan', 'PTV_PLAN']
# PTVboost = ['60 Boost PTV', 'boost ptv', 'boostPTV', 'boost PTV', 'BoostPTV', 'Boost PTV', 'BOOSTPTV', 
#            'Boost PTV_54', 'BoostPTV_57', 'Boost PTV60', 'Boost PTV 60', 'Boost PTV 63']
CTV = ['AdaptS1/S3CTV']

# Relabel the dataframe
# pd_PTV_data20000.loc[pd_PTV_data20000['original'].isin(PTVring), 'y'] = 'PTVring'
# pd_PTV_data20000.loc[pd_PTV_data20000['original'].isin(PTVplan), 'y'] = 'PTVplan'
# pd_PTV_data20000.loc[pd_PTV_data20000['original'].isin(PTVboost), 'y'] = 'PTVboost'
pd_PTV_data20000.loc[pd_PTV_data20000['original'].isin(CTV), 'y'] = 'CTV'



# In[25]:


# stack CTV and PTV together
pd_CTV_PTV_data20000 = pd.concat([pd_PTV_data20000, pd_CTV_data20000], axis = 0)

#Assign X and y
X = np.array(pd_CTV_PTV_data20000['ascii_length'])
X = np.array([lst for lst in X])
y = pd_CTV_PTV_data20000['y'].values

# X = pd_CTV_PTV_data20000['ascii_length'].to_list()
# y = pd_CTV_PTV_data20000['y'].values


# In[26]:


pd_CTV_PTV_data20000.to_csv('CTV_PTV_label1.csv', index = False)


# In[27]:


pd_CTV_PTV_data20000[pd_CTV_PTV_data20000['y'] == 'CTVp']


# ### Label CTV/PTV/GTV
# > Updated Jun 21, 202313, 14, 38, 29

# In[30]:


# define a function that can extract values from range
def val_range_contains(value, range_str):
    """
    check if the given value is within the range 
    """
    if isinstance(range_str, str) and '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return start <= value <= end
    else:
        return value == range_str


# In[31]:


CTV_clusters = [61, 110, 111, 113, 194, 282, 319, "349-496", 499, 500, 503, "522-525", 529, 
                531, 532, 624, "811-827", 857, 858, 984, 1000]


# In[32]:


# select the CTV clusters only
# make a copy so that we are not modifying the original 
pd_CTV_label = pd_data20000_label_name[pd_data20000_label_name['label'].apply(lambda x: any(val_range_contains(x, val) for val in CTV_clusters))].copy()
# add a column for y. Change all y to CTV first
pd_CTV_label.loc[:, 'y'] = 'CTV'


# In[42]:


ICTV = ['adaptictv', 'dnulargerictv', 'ictv', 'ictvmm', 'ictvzxl', 'ictva', 'ictvadapt', 'ictvall', 'ictvb',
       'ictvesophagus', 'ictvmd', 'ictvmediastinum', 'ictvmediastinum', 'ictvmfm', 'ictvmid', 'ictvnew', 
       'ictvold', 'ictvont', 'ictvplan', 'ictvtotal']
ICTVn = ['ictvnode', 'ictvnodes']
ICTVp = ['ictvp', 'ictvprimary']
IGTV = ['adaptigtv', 'dnulargerigtv']
PTV = ['adaptiveptv', 'adaptptvall', 'adapts/sptv']
CTVboost = ['boostctvforpa', 'boostctvtotal', 'ctvgyboost', 'ctvboost', 'ctvforboost', 'ctvboostgbg', 'ctvboostnew',
           'ctvbrainboost']
CTVcold = ['coldctv']
CTVring = ['ctvring', 'ctvringmm', 'ctvleftring', 'ctvringcm', 'ctvrightring', 'ctvtotalring',
          'ctvtotalringcm', 'ctvtotalringmm']
CTVn = ['ctvnodes', 'ctvnodalgbg', 'ctvnodeform', 'ctvnode', 'ctvnodal', 'ctvnodesdnu', 'ctvpelvicnodes', 
        'ctvplannode']
CTVp = ['ctvprim', 'ctvgbgprim', 'ctvprimaryd', 'ctvprimary', 'ctvpctv', 'ctvprimdef/', 'ctvpt']
CTV_GTV = ['dnuctvgtv']
GTV = ['dnuresgtvn', 'dnuresgtvn', 'adaptgtv']
GTVp = ['dnuresgtvp']

# delete upperjnx 17071, upperlip 17072
pd_CTV_label.drop([17071, 17072], inplace=True)

# Relabel the dataframe
pd_CTV_label.loc[pd_CTV_label['clean'].isin(GTV), 'y'] = 'GTV'
pd_CTV_label.loc[pd_CTV_label['clean'].isin(ICTV), 'y'] = 'ICTV'
pd_CTV_label.loc[pd_CTV_label['clean'].isin(ICTVn), 'y'] = 'ICTVn'
pd_CTV_label.loc[pd_CTV_label['clean'].isin(ICTVp), 'y'] = 'ICTVp'
pd_CTV_label.loc[pd_CTV_label['clean'].isin(IGTV), 'y'] = 'IGTV'
pd_CTV_label.loc[pd_CTV_label['clean'].isin(PTV), 'y'] = 'PTV'
pd_CTV_label.loc[pd_CTV_label['clean'].isin(CTVboost), 'y'] = 'CTVboost'
pd_CTV_label.loc[pd_CTV_label['clean'].isin(CTVcold), 'y'] = 'CTVcold'
pd_CTV_label.loc[pd_CTV_label['clean'].isin(CTVring), 'y'] = 'CTVring'
pd_CTV_label.loc[pd_CTV_label['clean'].isin(CTVn), 'y'] = 'CTVn'
pd_CTV_label.loc[pd_CTV_label['clean'].isin(CTVp), 'y'] = 'CTVp'
pd_CTV_label.loc[pd_CTV_label['clean'].isin(CTV_GTV), 'y'] = 'CTV_GTV'
pd_CTV_label.loc[pd_CTV_label['clean'].isin(GTVp), 'y'] = 'GTVp'


# In[128]:


PTV_clusters = [195, 242, 243, 285, 530, 863, 864, 1138, 1139, 1340, 1342, 1370, '1372-1387', 
                '1471-1534', 1602, '1975-1982', 2044, 2070, 2189, '2191-2195', 2418, 2419]
# select the PTV clusters only
# make a copy so that we are not modifying the original 
pd_PTV_label = pd_data20000_label_name[pd_data20000_label_name['label'].apply(lambda x: any(val_range_contains(x, val) for val in PTV_clusters))].copy()
# add a column for y. Change all y to CTV first
pd_PTV_label.loc[:, 'y'] = 'PTV'


# In[129]:


GTVboost = ['boostgtvgy']
# Delete = [1130, 1136]
PTVboost = ['boostptvmm', 'boostptvkc', 'bstptv', 'bstptvmm', 'bstptvcd', 'pptvboost', 'ptvboost', 'ptvboostmm']
PTVcold = ['coldptv']
IPTV = ['iptvmmeval','ptvi']
IPTVp = ['iptvp']
CTV = ['planningctv', 'ttotalctv', 'ttotalctv']
ITV = ['planningitv']
PTVp = ['pptv', 'ppaptv', 'pptvmm', 'pptvupper', 'pptvtarget', 'pptvcrop', 'pptvnew', 'pptvcge', 'pptvkc',
       'pptvlt', 'pptvrt', 'pptvg', 'pptvcord', 'pptvnew', 'pptvsup', 'pptvhot', 'pptvnewupp', 'pptvll',
       'pptvno', 'pptvrl', 'pptvsub', 'ptotalptv', 'ptvp', 'ptvprimary', 'tpptvupper', 'tpptv', 'tpptvnew',
       'tpptv', 'tpptvll', 'tpptvmax', 'tpptvmaxro', 'tpptvrl', 'tpptvrl/ll', 'tpptvrlnew', 'xpptv']
CTVp = ['ptotalctv']
PTVring = ['ptvcmring', 'ptvmmring', 'ptvringcm', 'ptvringmm', 'ptvrngmm', 'ringptv', 'xptvring', 
          'zptvring']
PTVn = ['ptvnodemm']
# what is stv?

# delete boost guide 1130, boost node 1136
pd_PTV_label.drop([1130, 1136], inplace=True)

# relabel
pd_PTV_label.loc[pd_PTV_label['clean'].isin(GTVboost), 'y'] = 'GTVboost'
pd_PTV_label.loc[pd_PTV_label['clean'].isin(PTVboost), 'y'] = 'PTVboost'
pd_PTV_label.loc[pd_PTV_label['clean'].isin(PTVcold), 'y'] = 'PTVcold'
pd_PTV_label.loc[pd_PTV_label['clean'].isin(IPTV), 'y'] = 'IPTV'
pd_PTV_label.loc[pd_PTV_label['clean'].isin(IPTVp), 'y'] = 'IPTVp'
pd_PTV_label.loc[pd_PTV_label['clean'].isin(CTV), 'y'] = 'CTV'
pd_PTV_label.loc[pd_PTV_label['clean'].isin(ITV), 'y'] = 'ITV'
pd_PTV_label.loc[pd_PTV_label['clean'].isin(PTVp), 'y'] = 'PTVp'
pd_PTV_label.loc[pd_PTV_label['clean'].isin(CTVp), 'y'] = 'CTVp'
pd_PTV_label.loc[pd_PTV_label['clean'].isin(PTVring), 'y'] = 'PTVring'
pd_PTV_label.loc[pd_PTV_label['clean'].isin(PTVn), 'y'] = 'PTVn'


# In[130]:


GTV_clusters = ['33-36', 40, 89, 284, '526-528', 632, '660-756', 796, '829-853', 986, 988, 1164, 1166, 
                1169, 1226, 1228]
# make a copy so that we are not modifying the original 
pd_GTV_label = pd_data20000_label_name[pd_data20000_label_name['label'].apply(lambda x: any(val_range_contains(x, val) for val in GTV_clusters))].copy()
# add a column for y. Change all y to CTV first
pd_GTV_label.loc[:, 'y'] = 'GTV'


# In[229]:


pd_GTV_label[940:950]


# In[230]:


GTVp = ['gtvpt', 'gtvptzxl', 'gtvprimary', 'gtvp', 'gtvpbh', 'gtvpbhbh', 'gtvphu', 'gtvpp', 'gtvprimary',
       'gtvprimarybed', 'gtvprimarycopy', 'gtvprimarydef', 'gtvprimarypost', 'gtvprimaryprec', 'gtvprimaryprei',
       'gtvprimprecht', 'gtvprmary', 'gtvptt', 'gtvptxyj', 'gtvrnodepostc', 'gtvrnodeprech']
IGTVn = ['igtvn', 'igtvnodes', 'igtvnodal', 'igtvnode']
IGTVp = ['igtvp', 'igtvprimary']
GTVcold = ['coldgtv']
GTVn = ['gtvnode', 'gtvn', 'gtvnt', 'gtvntjyc', 'gtvntt', 'gtvntxyj', 'gtvntzxl', 'gtvnbh', 'gtvnbhbh',
       'gtvnhu', 'gtvnmip', 'gtvndef', 'gtvngbg', 'gtvnleft', 'gtvnodal', 'gtvnodalnewscn', 'gtvnodalprechemo',
       'gtvnode', 'gtvnode?', 'gtvnodecopy', 'gtvnodem', 'gtvnodepostc', 'gtvnodeprech', 'gtvnodepet', 
       'gtvnode/', 'gtvnodebed', 'gtvnodebstres', 'gtvnodeece', 'gtvnodeexpand', 'gtvnodefnaneg', 'gtvnodegbg',
       'gtvnodeinterme', 'gtvnodepetleft', 'gtvnodepostcht', 'gtvnodepostin', 'gtvnodeprechem', 'gtvnodeprecht',
       'gtvnodepreindu', 'gtvnodepremfm', 'gtvnodepreop', 'gtvnoderightneck', 'gtvnodes', 'gtvnodescopy', 
       'gtvnodet', 'gtvnpre', 'gtvnprechemo', 'gtvnright', 'gtvprechtnodal', 'gtvprechemonod', 'nodalgtv',
       'nodegtv', 'nodesgtv']
GTVboost = ['gtvboost', 'gtvboostgy', 'gtvnt', 'gtvntjyc', 'gtvntt', 'gtvntxyj', 'gtvntzxl', 'lnboostgtv']
GTVpn = ['gtvprimarynode', 'gtvprimnodes']
GTVring = ['gtvring']
IGTV = ['huigtv', 'igtv', 'igtvhu', 'igtvmm', 'igtvgy', 'igtvadapt', 'igtvadjhu', 'igtvall', 'igtvboost', 
       'igtvcopy', 'igtvesophagus', 'igtvhucorr', 'igtvhucorrct', 'igtvhucorrect', 'igtvhucrrect', 
       'igtvhuoriginal', 'igtvhuoverride', 'igtviitvmm', 'igtvinlung', 'igtvln', 'igtvlns', 'igtvlul',
       'igtvmd', 'igtvmediastinum', 'igtvnohu', 'igtvoverride', 'igtvrightscv', 'igtvtotal', 'igtvzl', 
       'originaligtv']
CTVn = ['nodesctv']
ICTV = ['origictv', 'originalictv']

# relabel
pd_GTV_label.loc[pd_GTV_label['clean'].isin(GTVn), 'y'] = 'GTVn'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(GTVp), 'y'] = 'GTVp'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(IGTVn), 'y'] = 'IGTVn'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(IGTVp), 'y'] = 'IGTVp'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(GTVcold), 'y'] = 'GTVcold'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(GTVboost), 'y'] = 'GTVboost'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(GTVpn), 'y'] = 'GTVpn'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(GTVring), 'y'] = 'GTVring'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(IGTV), 'y'] = 'IGTV'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(CTVn), 'y'] = 'CTVn'
pd_GTV_label.loc[pd_GTV_label['clean'].isin(ICTV), 'y'] = 'ICTV'


# In[235]:


# Stack CTV, PTV, GTV dataframe together
pd_CTV_PTV_GTV = pd.concat([pd_CTV_label, pd_PTV_label, pd_GTV_label], axis=0)



# ### OneHot Encoding
# > Started Jun 16, 2023

# In[28]:


# Build Char_dict
CHAR_DICT = 'abcdefghijklmnopqrstuvwxyz'


# In[238]:


def build_char_dictionary(char_dict=None, unknown_label='UNK'):
        """
            Define possbile char set. Using "UNK" if character does not exist in this set
        """ 
        
        if char_dict is None:
            char_dict = CHAR_DICT


        chars = []

        for c in char_dict:
            chars.append(c)

        chars = list(set(chars))
        
        chars.insert(0, unknown_label)

        num_of_char = len(chars)
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))
        
        print('Totoal number of chars:', num_of_char)

        print('First 3 char_indices sample:', {k: char_indices[k] for k in list(char_indices)[:3]})
        print('First 3 indices_char sample:', {k: indices_char[k] for k in list(indices_char)[:3]})
        
        return char_indices, indices_char, num_of_char


# In[239]:

# Be careful with this. Every time different dicts will be generated
# Better generate once and save the dictionaries

# char_indices, indices_char, num_of_char = build_char_dictionary(char_dict=None, unknown_label='UNK')


# In[240]:


print(char_indices)


# In[29]:


char_indices = {'UNK': 0, 'z': 1, 'j': 2, 'm': 3, 'y': 4, 'd': 5, 'e': 6, 'q': 7, 'f': 8, 'k': 9, 'l': 10, 'o': 11, 's': 12, 'r': 13, 'c': 14, 'h': 15, 'b': 16, 'w': 17, 'i': 18, 'g': 19, 'p': 20, 't': 21, 'v': 22, 'n': 23, 'x': 24, 'a': 25, 'u': 26}


# In[305]:


# Save this dictionary locally so that it doesn't change every time
with open('char_dict.pickle', 'wb') as file:
    # wb specifies it's being opened in binary format for writing
    pickle.dump(char_indices, file)


# In[30]:


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


# In[31]:


# # convert string labels to numeric
CTV_PTV_labels = pd_CTV_PTV_data20000['y'].unique()
CTV_PTV_label2indexes, CTV_PTV_index2labels = convert_labels(CTV_PTV_labels)

# # for CTV/PTV/GTV
# CTV_PTV_GTV_labels = pd_CTV_PTV_GTV['y'].unique()
# CTV_PTV_GTV_label2indexes, CTV_PTV_GTV_index2labels = convert_labels(CTV_PTV_GTV_labels)


# In[32]:


preprocess_x = pd_CTV_PTV_data20000['clean'].values
preprocess_y = pd_CTV_PTV_data20000['y'].values
# preprocess_x = pd_CTV_PTV_GTV['clean'].values
# preprocess_y = pd_CTV_PTV_GTV['y'].values


# In[33]:


# onehot encoding, record number of occurances 
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


# In[34]:


def onehot_encode2(x_raw, char_dict):
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


# In[35]:


x_encode1, y_encode1 = onehot_encode1(preprocess_x, preprocess_y, char_indices, CTV_PTV_label2indexes)


# ### Tensorflow
# > Started Jun 13, 2023


# #### Trial 3 with OneHot
# > Updated Jun 16, 2023

# In[ ]:


# there are nine labels 


# In[36]:


X_train_oh, X_test_oh, y_train_oh, y_test_oh = train_test_split(x_encode1, y_encode1, test_size = 0.5, stratify=y_encode1, random_state=1)




# In[37]:


# Define the input shape
tf.keras.backend.clear_session()
input_shape = [27]

# Create the model
model = tf.keras.models.Sequential()

# Add the input layer
model.add(tf.keras.layers.Input(shape=input_shape))

# Add other layers to the model

model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
model.add(tf.keras.layers.Dense(5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()


# In[63]:


oh_history = model.fit(X_train_oh, y_train_oh, validation_data=(X_test_oh, y_test_oh), epochs=300)


# In[64]:


# Access the training and validation loss
train_loss = oh_history.history['loss']
val_loss = oh_history.history['val_loss']


# In[324]:


# Create a range of epochs
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



# In[325]:


train_accuracy = oh_history.history['accuracy']
val_accuracy = oh_history.history['val_accuracy']


# In[326]:


# Create a range of epochs
epochs = range(1, len(train_accuracy) + 1)
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



# In[65]:


# Confusion matrix
y_pred_oh = model.predict(X_test_oh)
y_pred_oh = np.argmax(y_pred_oh, axis=1)
cm = tf.math.confusion_matrix(y_test_oh, y_pred_oh)



# In[69]:


plt.imshow(cm, cmap='viridis')
plt.colorbar()
plt.show()


# In[334]:


probabilities = model.predict(X_test_oh)



# In[329]:


probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


# In[54]:


predictions_oh = probability_model.predict(X_test_oh)




# In[55]:


result = np.apply_along_axis(np.argmax, axis=1, arr=predictions_oh)


# #### Plot Probabilities

# In[70]:


# Find the data for each class 
PTV_index = np.where(y_test_oh == 0)
PTV_data = X_test_oh[PTV_index[0]]

CTV_index = np.where(y_test_oh == 1)
CTV_data = X_test_oh[CTV_index[0]]

ICTV_index = np.where(y_test_oh == 2)
ICTV_data = X_test_oh[ICTV_index[0]]

CTVn_index = np.where(y_test_oh == 3)
CTVn_data = X_test_oh[CTVn_index[0]]

CTVp_index = np.where(y_test_oh == 4)
CTVp_data = X_test_oh[CTVp_index[0]]




# In[72]:


Probabilities_PTV = model.predict(PTV_data)
Probabilities_CTV = model.predict(CTV_data)
Probabilities_ICTV = model.predict(ICTV_data)
Probabilities_CTVn = model.predict(CTVn_data)
Probabilities_CTVp = model.predict(CTVp_data )




# In[79]:


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


plt.savefig('prob_distri_test.png')

plt.show()



# #### Save the model 

# In[340]:


get_ipython().system('mkdir -p saved_model')
model.save('saved_model/model1')


# #### Predict

# In[280]:


x_predict_preprocess = pd_PTV_label['clean'].values


# In[289]:


x_predict = onehot_encode2(x_predict_preprocess, char_indices)


# In[293]:


test_x = x_predict[10:20]



# In[407]:


prediction = model.predict(x_predict)
result = np.apply_along_axis(np.argmax, axis=1, arr=prediction)


