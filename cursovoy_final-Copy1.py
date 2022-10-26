#!/usr/bin/env python
# -*- coding: utf-8 -*-     ## coding: utf-8

# ## Курсовой проект 
# 
# ### Задание для курсового проекта
# 
# Метрика:
# R2 - коэффициент детерминации (sklearn.metrics.r2_score)
# 
# Сдача проекта:
# 1. Прислать в раздел Задания Урока 10 ("Вебинар. Консультация по итоговому проекту")
# ссылку на программу в github (программа должна содержаться в файле Jupyter Notebook 
# с расширением ipynb). (Pull request не нужен, только ссылка ведущая на сам скрипт).
# 2. Приложить файл с названием по образцу SShirkin_predictions.csv
# с предсказанными ценами для квартир из test.csv (файл должен содержать два поля: Id, Price).
# В файле с предсказаниями должна быть 5001 строка (шапка + 5000 предсказаний).
# 
# Сроки и условия сдачи:
# Дедлайн: сдать проект нужно в течение 72 часов после начала Урока 10 ("Вебинар. Консультация по итоговому проекту").
# Для успешной сдачи должны быть все предсказания (для 5000 квартир) и R2 должен быть больше 0.6.
# При сдаче до дедлайна результат проекта может попасть в топ лучших результатов.
# Повторная сдача и проверка результатов возможны только при условии предыдущей неуспешной сдачи.
# Успешный проект нельзя пересдать в целях повышения результата.
# Проекты, сданные после дедлайна или сданные повторно, не попадают в топ лучших результатов, но можно узнать результат.
# В качестве итогового результата берется первый успешный результат, последующие успешные результаты не учитываются.
# 
# Примечание:
# Все файлы csv должны содержать названия полей (header - то есть "шапку"),
# разделитель - запятая. В файлах не должны содержаться индексы из датафрейма.
# 
# Рекомендации для файла с кодом (ipynb):
# 1. Файл должен содержать заголовки и комментарии
# 2. Повторяющиеся операции лучше оформлять в виде функций
# 3. Не делать вывод большого количества строк таблиц (5-10 достаточно)
# 4. По возможности добавлять графики, описывающие данные (около 3-5)
# 5. Добавлять только лучшую модель, то есть не включать в код все варианты решения проекта
# 6. Скрипт проекта должен отрабатывать от начала и до конца (от загрузки данных до выгрузки предсказаний)
# 7. Весь проект должен быть в одном скрипте (файл ipynb).
# 8. При использовании статистик (среднее, медиана и т.д.) в качестве признаков,
# лучше считать их на трейне, и потом на валидационных и тестовых данных не считать 
# статистики заново, а брать их с трейна. Если хватает знаний, можно использовать кросс-валидацию,
# но для сдачи этого проекта достаточно разбить данные из train.csv на train и valid.
# 9. Проект должен полностью отрабатывать за разумное время (не больше 10 минут),
# поэтому в финальный вариант лучше не включать GridSearch с перебором 
# большого количества сочетаний параметров.
# 10. Допускается применение моделей машинного обучения из библиотеки sklearn.

# ### Подключение библиотек и скриптов

# In[1]:


import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import KFold, GridSearchCV


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
import lightgbm
from lightgbm import LGBMRegressor


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


matplotlib.rcParams.update({'font.size': 14})


# In[4]:


def evaluate_preds(train_true_values, train_pred_values, test_true_values, test_pred_values):
  
   print("Train R2:\t" + str(round(r2(train_true_values, train_pred_values), 3)))
   print("Test R2:\t" + str(round(r2(test_true_values, test_pred_values), 3)))

   plt.figure(figsize=(18,10))
   plt.subplot(121)
   sns.scatterplot(x=train_pred_values, y=train_true_values)
   plt.plot([0, 500000], [0, 500000], linestyle='--', color='black')
   plt.xlabel('Predicted values')
   plt.ylabel('True values')
   plt.title('Train sample prediction')
   
   plt.subplot(122)
   sns.scatterplot(x=test_pred_values, y=test_true_values)
   plt.plot([0, 500000], [0, 500000], linestyle='--', color='black')
   plt.xlabel('Predicted values')
   plt.ylabel('True values')
   plt.title('Test sample prediction')

   plt.show()


# ### Пути к директориям и файлам

# In[5]:


TRAIN_DATASET_PATH = 'project_task/train.csv' # y_train, y_valid
TEST_DATASET_PATH = 'project_task/test.csv'


# **Описание датасета**
# 
# * **Id** - идентификационный номер квартиры
# * **DistrictId** - идентификационный номер района
# * **Rooms** - количество комнат
# * **Square** - площадь
# * **LifeSquare** - жилая площадь
# * **KitchenSquare** - площадь кухни
# * **Floor** - этаж
# * **HouseFloor** - количество этажей в доме
# * **HouseYear** - год постройки дома
# * **Ecology_1, Ecology_2, Ecology_3** - экологические показатели местности
# * **Social_1, Social_2, Social_3** - социальные показатели местности
# * **Healthcare_1, Helthcare_2** - показатели местности, связанные с охраной здоровья
# * **Shops_1, Shops_2** - показатели, связанные с наличием магазинов, торговых центров
# * **Price** - цена квартиры

# In[6]:


train_df = pd.read_csv(TRAIN_DATASET_PATH)
train_df.tail()


# In[7]:


test_df = pd.read_csv(TEST_DATASET_PATH)
test_df.tail()


# In[8]:


train_df.columns


# ### Приведение типов

# In[9]:


train_df.dtypes


# In[10]:


train_df.dtypes.value_counts()


# ### Разбиение на train и valid

# In[11]:


feature_names = ['Id','DistrictId','Rooms', 'Square', 'LifeSquare', 'KitchenSquare', 'Floor', 'HouseFloor', 'HouseYear',
                 'Ecology_1', 'Ecology_2', 'Ecology_3', 'Social_1', 'Social_2', 'Social_3',
                 'Healthcare_1', 'Helthcare_2', 'Shops_1', 'Shops_2', 'Price']

target_name = 'Price'


# In[12]:


X = train_df[feature_names]
y = train_df[target_name]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=21) # test_size=0.33 ????
X_valid = X_valid.drop('Price', axis=1)


# In[13]:


X_train.shape


# In[14]:


X_valid.shape


# In[15]:


X_valid.describe()


# In[16]:


X_train.dtypes


# In[17]:


X_train.dtypes.value_counts()


# In[18]:


X_train['Id'] = X_train['Id'].astype(str)
X_train['DistrictId'] = X_train['DistrictId'].astype(str)


# ## Обзор данных

# ### Целевая переменная

# In[19]:


plt.figure(figsize = (16, 8))

#X_train['Price'].hist(bins=30)
train_df[target_name].hist(bins=30)
plt.ylabel('Count')
plt.xlabel('Price')

plt.title('Target distribution')
plt.show()


# ### Количественные переменные

# In[20]:


X_train.describe()


# In[21]:


X_valid.describe()


# ### Категориальные признаки

# In[22]:


def cat_features(df):
    cat_colnames = df.select_dtypes(include='object').columns.tolist()
    for cat_colname in cat_colnames[2:]:
        print(str(cat_colname) + '\n\n' + str(X_train[cat_colname].value_counts()) + '\n' + '*' * 100 + '\n')
    
    
    return df


# ### Обработка выбросов

# In[23]:


def df_fix(df):
    df.loc[df['Rooms'].isin([0, 10, 19]), 'Rooms'] = df['Rooms'].median()
    df.loc[df['Square'] > df['Square'].quantile(.99), 'Square'] = df['Square'].median()  
    df.loc[df['Square'] < df['Square'].quantile(.01), 'Square'] = df['Square'].median()
    df.loc[df['LifeSquare'].isna(), 'LifeSquare'] = df['LifeSquare'].median()
    df.loc[df['LifeSquare'] > df['LifeSquare'].quantile(.99), 'LifeSquare'] = df['LifeSquare'].median()
    df.loc[df ['LifeSquare'] < df['LifeSquare'].quantile(.01), 'LifeSquare'] = df['LifeSquare'].median()
    df.loc[X_train['KitchenSquare'].isnull(), 'KitchenSquare'] = df['KitchenSquare'].median
    df.loc[X_train['KitchenSquare'] < 3, 'KitchenSquare'] = 3
    df.loc[X_train['KitchenSquare'] > 25, 'KitchenSquare'] = df['KitchenSquare'].median()

    df.loc[df['Square']< df["KitchenSquare"] + df["LifeSquare"], 'Square'] =  df["KitchenSquare"] + df["LifeSquare"]
    df.loc[df['HouseFloor'] == 0, 'HouseFloor'] = df['HouseFloor'].median()
    floor_outliers = df[X_train['Floor'] > df['HouseFloor']].index
    df.loc[floor_outliers, 'Floor'] = df.loc[floor_outliers, 'HouseFloor'].apply(lambda x: random.randint(1, x))
    df.loc[X_train['HouseYear'] > 2020, 'HouseYear'] = 2020
    df.loc[X_train['Healthcare_1'].isnull(), 'Healthcare_1'] = df['Healthcare_1'].median()
    
    return df
    


# In[24]:


df_fix(X_train)


# In[25]:


X_train.describe()


# In[26]:


X_train['Square'].sort_values(ascending=False)


# In[27]:


X_train['LifeSquare'].sort_values(ascending=False)


# In[28]:


X_train['KitchenSquare'].sort_values(ascending=False)


# In[29]:


X_train['HouseFloor'].sort_values().unique()


# In[30]:


X_train['Floor'].sort_values().unique()


# In[31]:


X_train['HouseYear'].value_counts().sort_index()


# In[32]:


X_train.shape


# ### Обработка пропусков

# In[33]:


def new_cat_features(df):
    df['Ecology_2_bin'] = df['Ecology_2'].replace({'A':0, 'B':1})
    df['Ecology_3_bin'] = df['Ecology_3'].replace({'A':0, 'B':1})
    df['Shops_2_bin'] = df['Shops_2'].replace({'A':0, 'B':1})
    
 
    
    return df


# In[34]:


new_cat_features(X_train)


# In[35]:


X_train.shape


# In[36]:


X_train.drop(['Ecology_2', 'Ecology_3', 'Shops_2'], axis=1, inplace = True)


# In[37]:


X_train.shape


# In[38]:


len(X_train) - X_train.count()


# In[39]:


X_train.info()


# ## Построение новых признаков (feature engineering)

# ### DistrictSize

# In[40]:


district_size = X_train['DistrictId'].value_counts().reset_index()                          .rename(columns={'index':'DistrictId','DistrictId':'DistrictSize'})


# In[41]:


X_train = X_train.merge(district_size, on=['DistrictId'], how='left')
X_train.head()


# ### MedPricePerSqmInDr

# In[42]:


X_train['Square'] = X_train['Square'].astype(float)


# In[43]:


X_train.head()


# In[44]:


info_by_district_id = X_train.groupby(['DistrictId', 'Rooms',], as_index=False).agg({'Price':'median', 'Square': 'median'})                        .rename(columns={'Price':'MedPriceByDistrict', 'Square': 'MedSquareByDistrict'})


# In[45]:


info_by_district_id['MedPricePerSqmInDr'] = info_by_district_id['MedPriceByDistrict']         / info_by_district_id['MedSquareByDistrict']
info_by_district_id.drop(['MedPriceByDistrict', 'MedSquareByDistrict'], axis=1, inplace=True)


# In[46]:


X_train = X_train.merge(info_by_district_id, on=['DistrictId', 'Rooms'], how='left')

X_train['MedPricePerSqmInDr'] = X_train['MedPricePerSqmInDr'].fillna(
        X_train['MedPricePerSqmInDr'].mean())

X_train.head()


# In[47]:


X_train.shape


# In[48]:


X_train.columns.tolist()


# ### Анализ признаков

# In[49]:


df_num_features = X_train.select_dtypes(include=['float64'])
df_num_features.head(n=2)


# In[50]:


df_num_features.hist(figsize=(16,16), bins=20, grid =True);


# ## Обработка валидационной выборки
# 

# In[51]:


def valid_fix(df):
    df.loc[df['Rooms'].isin([0, 19]), 'Rooms'] = df['Rooms'].median()
    df.loc[df['Square'] > df['Square'].quantile(.99), 'Square'] = df['Square'].median()  
    df.loc[df['Square'] < df['Square'].quantile(.01), 'Square'] = df['Square'].median()
   
    df.loc[df['LifeSquare'].isna(), 'LifeSquare'] = df['LifeSquare'].median()
    df.loc[df['LifeSquare'] > df['LifeSquare'].quantile(.99), 'LifeSquare'] = df['LifeSquare'].median()
    df.loc[df ['LifeSquare'] < df['LifeSquare'].quantile(.01), 'LifeSquare'] = df['LifeSquare'].median()
    df.loc[df['KitchenSquare'].isnull(), 'KitchenSquare'] = df['KitchenSquare'].median
    
    
    df.loc[df['KitchenSquare'].isnull(), 'KitchenSquare'] = df['KitchenSquare'].median
    df.loc[df['KitchenSquare'] < 3, 'KitchenSquare'] = 3
    df.loc[df['KitchenSquare'] > 25, 'KitchenSquare'] = df['KitchenSquare'].median()
    

    df.loc[df['Square']< df["KitchenSquare"] + df["LifeSquare"], 'Square'] =  df["KitchenSquare"] + df["LifeSquare"]
    
    
    df.loc[df['HouseFloor'] == 0, 'HouseFloor'] = df['HouseFloor'].median()
    floor_outliers = df[df['Floor'] > df['HouseFloor']].index
    df.loc[floor_outliers, 'Floor'] = df.loc[floor_outliers, 'HouseFloor'].apply(lambda x: random.randint(1, x))
    
    df.loc[df['HouseYear'] > 2020, 'HouseYear'] = 2020
    df.loc[df['Healthcare_1'].isnull(), 'Healthcare_1'] = df['Healthcare_1'].median()
    
    return df


# In[52]:


X_valid.describe()


# In[53]:


valid_fix(X_valid)


# In[54]:


new_cat_features(X_valid)


# In[55]:


X_valid.drop(['Ecology_2', 'Ecology_3', 'Shops_2'], axis=1, inplace = True)


# In[56]:


X_valid.head()


# In[57]:


X_valid['DistrictId'] = X_valid['DistrictId'].astype(str)


# In[58]:


district_size_v = X_valid['DistrictId'].value_counts().reset_index()                          .rename(columns={'index':'DistrictId','DistrictId':'DistrictSize'})


# In[59]:


X_valid = X_valid.merge(district_size_v, on=['DistrictId'], how='left')
X_valid.head()


# ### Перенос признака MedPricePerSqmInDr на валидационную выборку

# In[60]:



X_valid = X_valid.merge(info_by_district_id, on=['DistrictId', 'Rooms'], how='left')

X_valid['MedPricePerSqmInDr'] = X_valid['MedPricePerSqmInDr'].fillna(
        X_valid['MedPricePerSqmInDr'].mean())

X_valid.head()


# In[61]:


X_train.drop(['Price'], axis=1, inplace = True)


# In[62]:


X_valid.head()


# In[63]:


X_valid.shape


# In[64]:


X_valid.isnull().sum()


# ## Обучение и оценка модели

# In[65]:


X_train.shape


# In[66]:


y_train.shape


# In[67]:


X_valid.shape


# In[84]:


lgbm_model = LGBMRegressor(max_depth=2, n_estimators=200, random_state=21)
lgbm_model.fit(X_train.values, y_train)


# In[85]:


y_train_preds = lgbm_model.predict(X_train.values)
y_valid_preds = lgbm_model.predict(X_valid.values)

evaluate_preds(y_train, y_train_preds, y_valid, y_valid_preds)


# ### Важность признаков

# In[70]:


feature_importances = pd.DataFrame(zip(X_train.columns, gb_model.feature_importances_), 
                                   columns=['feature_name', 'importance'])

feature_importances.sort_values(by='importance', ascending=False)


# ## Тестовая выборка

# In[71]:


test_df.describe()


# In[72]:


valid_fix(test_df)


# In[73]:


new_cat_features(test_df)


# In[74]:


test_df.drop(['Ecology_2', 'Ecology_3', 'Shops_2'], axis=1, inplace = True)


# In[75]:


test_df.head()


# In[76]:


test_df['DistrictId'] = test_df['DistrictId'].astype(str)


# In[77]:


district_size_t = test_df['DistrictId'].value_counts().reset_index()                          .rename(columns={'index':'DistrictId','DistrictId':'DistrictSize'})


# In[78]:


test_df = test_df.merge(district_size_t, on=['DistrictId'], how='left')
test_df.head()


# ### Перенос признака MedPricePerSqmInDr на тестовую выборку

# In[79]:


test_df = test_df.merge(info_by_district_id, on=['DistrictId', 'Rooms'], how='left')

test_df['MedPricePerSqmInDr'] = test_df['MedPricePerSqmInDr'].fillna(
        test_df['MedPricePerSqmInDr'].mean())

test_df.head()


# In[80]:


test_df.shape


# In[81]:


test_df.isnull().sum()


# In[82]:


X_train.shape


# ### Формирование файла csv с прогнозами

# In[87]:


test_id = test_df["Id"]
pred_df = pd.DataFrame()
pred_df["Id"] = test_id
pred_df["Price"] = lgbm_model.predict(test_df.values)
assert pred_df.shape[0] == 5000, f"Real pred-shape = {pred_df.shape[0]}, Expected pred-shape = 5000"

pred_df[['Price']].round(2).to_csv('./predictions.csv', index=None)


# In[ ]:




