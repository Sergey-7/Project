## Разработка модели для сегментации объектов на планах помещений по изображениям

Цель проекта регионального чемпионата Цифровой прорыв - разработать алгоритм автоматического распознавания планов помещений путем создания моделей семантической сегментации, принимающих на вход растровые изображения, содержащие планы жилого помещения. Для обучения модели, осуществляющей стандартизацию формата начертания плана, использовался датасет, разработанный методом кроссвалидационной ручной разметки. Для сегментации стен и окон была обучена нейросеть DeepLabV3+; для детекции дверей использовалась сеть fasterrcnn_resnet50_fpn.

Метрика качества: MeanAveragePrecision, binary_accuracy, Diceloss.

Использованные инструменты и библиотеки: Python, Tensorflow, Pytorch, DeepLabV3+

https://github.com/Sergey-7/Chempionat

## Исследование данных о продаже квартир

### Курсовой проект для курса "Python для Data Science" 

Цель проекта - проанализировать данные о недвижимости, чтобы определить, как влияют определенные факторы на целевую переменную Price -стоимость недвижимости.        Данные обучающего датасета содержат такие признаки, как количество комнат, площадь, этажность, год постройки, социальные и экологические показатели. На основе данных после проведения EDA, очистки и обработки данных строится модель для предсказания цен на недвижимость (квартиры). Полученная модель позволяет предсказывать цены для квартир из тестового датасета. 

Метрика качества: R2 - коэффициент детерминации (sklearn.metrics.r2_score).

Использованные инструменты и библиотеки: Pandas, numpy, sklearn, matplotlib, Lightgbm

https://github.com/Sergey-7/Libr/blob/%D0%A1urs_project/Cursovoy_project/cursovoy_final.ipynb

## Исследование надежности заемщиков

### Курсовой проект для курса "Библиотеки Python для Data Science: продолжение "  

Цель проекта - проанализировать данные о клиентах банка, чтобы спрогнозировать невыполнение долговых обязательств по текущему кредиту. Данные содержат статистику о платежеспособности клиентов по кредитам. На основе данных после проведения EDA, очистки и обработки данных построена модель для прогнозирования клиентами банка невыполнения долговых обязательств по текущему кредиту. 

Метрика качества — F1-score (sklearn.metrics.f1_score). 

Использованные инструменты и библиотеки: Pandas, numpy, sklearn, scipy, matplotlib, seaborn, Catboost. 

https://github.com/Sergey-7/Libr/blob/CP_2/C_project/Cursovoy-final.ipynb


## Создание модели БД MYSQL интернет-магазина

### Курсовой проект по курсу «Основы реляционных баз данных. MYSQL». 

 
Цель проекта — спроектировать модель БД для хранения данных интернет-магазина. База данных включает 10 таблиц: каталог товаров, категории товаров, бренды, покупатели, анкеты покупателей (profiles), товары , изменение цены на товары, заказы, скидки, проданные товары. БД позволяет помимо хранения всех внесенных данных вести учет купленных товаров (выполненных заказов), менять цены на отдельные товары, назначать скидки на конкретный товар для конкретного покупателя.

Созданы скрипты структуры БД; ERDiagram для БД; скрипты наполнения БД данными; скрипты характерных выборок ; представления; хранимые процедуры / триггеры.  

Также прилагается скрин файла в формате ipynb (для Jupiter Notebook) как пример чтения данных из данной бд 
путем сложного запроса и преобразования их в таблицу DataFrame, которую можно использовать для обработки и подготовки данных
для дальнейшего анализа и построения моделей Data Science - http://joxi.ru/KAxdnkysZNgMRA

https://github.com/Sergey-7/MYSQL/tree/master/Cursovoy

## Прогнозирования цен на недвижимость

### Тренировочные соревнования на Kaggle (уровень Get started) House Prices - Advanced Regression Techniques    

Цель проекта — использование продвинутых алгоритмов обучения для прогнозирования цен на жилье. Обучающие данные содержат более 80 признаков. После очистки и преобразования данных построена модель для прогнозирования цен на недвижимость. 

Использованные инструменты: Python, Pandas, Matplotlib, Numpy, Scikit, Skipy, StatsModels, Seaborn, Keras, TensorFlow.

https://github.com/Sergey-7/Kaggle_projects/blob/Project_1/House_test.ipynb

## Прогнозирования продаж товаров различных категорий

### Тренировочные соревнования на Kaggle (уровень Get started)  Store Sales - Time Series Forecasting    

Цель проекта — предсказать продажи большого количества товаров разных категорий. Данные обучающего датасета включают даты продаж, магазины и информацию о товарах, данные о рекламе продаваемых товаров, а также количество продаж. 

Использованные инструменты: Python, Pandas, Matplotlib, Numpy, Scikit, Skipy, StatsModels, Seaborn, Keras, TensorFlow.

https://github.com/Sergey-7/Kaggle_2/blob/ff83f1fb87e22507b32da1c8a801252fbfdd0ec1/Project_2/store-sales-using-a-hybrid-model.ipynb

## Исследование надежности заемщиков по данным кредитных историй клиентов

###  Соревнование на ODS.aiй.  

Цель проекта - на основе кредитной истории клиента нужно оценить, насколько благонадежным является клиент, и определить вероятность его ухода в дефолт по новому кредиту. Каждый кредит описывается набором из 60 категориальных признаков. В проекте используются огромные по объемпмм выборки (логи транзакций клиентов), и поэтому для эффективного чтения данных, выделения признаков и построения решений на локальных машинах с большими ограничениями по RAM используются инструменты для работы с форматом данных Parquet. 

Целевая метрика - ROC AUC.

https://colab.research.google.com/drive/1fPhwdmvqgxzl8S0cSNETWRcmP9fR_mXs?usp=sharing


## Исследование надежности заемщиков по данным карточных транзакций клиентов

###  Соревнование на ODS.aiй.  

Цель проекта - на основе истории транзакций клиентов банка оценить насколько благонадежным является клиент и определить шансы того, что он уйдет в дефолт. В проекте используются громные выборки (логи транзакций клиентов), и поэтому для эффективного чтения данных, выделения признаков и построения решений на локальных машинах с большими ограничениями по RAM используются инструменты для работы с форматом данных Parquet. 

Целевая метрика - ROC AUC.

https://colab.research.google.com/drive/1JryGt155e2CyzyS_I8dCdkirpVv4OJgA?usp=sharing

## Парсинг данных с вебресурса и их визуализация в Gephi

### Пет-проект

Цель проекта - парсинг данных по рецептам салатов с сайта 'https://www.allrecipes.com/recipes/96/salad/' и запись выбранных параметров в файлы csv. Спарсенные данные преобразуются в узлы (данные по калорийности салатов) и ребра ( данные по совпадению используемых в них ингредиентов) визуализируемых графов в программе Gephi, описывающих соотношения и сравнение разных рецептов по калорийности и ингредиентам. Чем толще ребра между узлами, тем больше одинаковых ингредиентов в рецептах, а чем толще узел, то рецепт более калорийный. Прилагается скрин рабочей области Gephi 0.9.2.

https://github.com/Sergey-7/Project/tree/master/Gephi_%26_parsing

## Создание интернет-магазина в фреймворке Django

Цель проекта - создать учебный сайт интернет-магазина во фреймворке Django. Используемые инструменты: Django 2.1, Python 3.6, SQLite.

https://github.com/Sergey-7/mygeekshop
