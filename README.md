EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
       data
<img width="1350" height="759" alt="499480440-c53c0b6a-b202-4e07-ba8d-bc92f42a99c0" src="https://github.com/user-attachments/assets/b6de0f77-df7f-4ac5-b6cb-a900b61243fb" />

       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       data.isnull().sum()
<img width="390" height="335" alt="499481459-7fe1e04f-4f66-4911-9b4e-4a119e1c70af" src="https://github.com/user-attachments/assets/7fc57e60-53e0-41d9-a5e4-fdd8c281a7ad" />

       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       missing=data[data.isnull()]
       missing
<img width="1291" height="444" alt="499482087-83f16370-85ef-489b-8310-53428dc52c38" src="https://github.com/user-attachments/assets/e88795f3-d887-41a1-99e6-0b5846912d06" />

       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       missing=data[data.isnull().any(axis=1)]
       missing


<img width="1247" height="63" alt="499482692-4f21859f-fbcf-41a4-9271-3a39b6da41a7" src="https://github.com/user-attachments/assets/75c1a888-7653-4756-b292-d857ccb18a78" />

       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       data2=data.dropna(axis=0)
       data2

<img width="1291" height="736" alt="499483209-e12bc2f8-fc8a-4a3a-9ef1-7e288009af6c" src="https://github.com/user-attachments/assets/590c8d3e-01ca-4ac7-b1a7-2f56bbc0f8e0" />

       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       sal=data["SalStat"]
       data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1}
       print(data2['SalStat'])

<img width="497" height="262" alt="499483936-ec902af9-5e5f-4967-aa6d-a67ce87bc982" src="https://github.com/user-attachments/assets/d805e7e8-6b6c-4059-b91a-2b7705239a26" />

       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       data2

<img width="1343" height="579" alt="499484382-484fd890-6762-43cb-80c2-6e9f6c824d0d" src="https://github.com/user-attachments/assets/d7f190b3-fa25-4baa-bccc-ccd31dd02e0a" />


       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       data2
       new_data=pd.get_dummies(data2, drop_first=True)
       new_data       

<img width="1343" height="536" alt="499484971-d71a7265-752d-4d8e-bd13-f0ee04543dce" src="https://github.com/user-attachments/assets/ab9ab0fa-833a-4229-8a39-649c634ab90b" />

       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       columns_list=list(new_data.columns)
       print(columns_list)

<img width="1261" height="450" alt="499485351-54aeb1e2-eacc-46f0-9b5b-d495f7bbac42" src="https://github.com/user-attachments/assets/8ce150d8-46c8-4eec-9a8c-1fcc86802779" />

       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       features=list(set(columns_list)-set(['SalStat']))
       print(features)

<img width="1266" height="456" alt="499485655-a3e081b0-5b6e-4ad0-8644-5fa3add47d49" src="https://github.com/user-attachments/assets/65943a63-6757-4dc4-bdb6-a762b578e9bc" />

       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       y=new_data['SalStat'].values

<img width="255" height="53" alt="499486023-ec16c168-0da7-468d-9fd0-bf80fa8b89ac" src="https://github.com/user-attachments/assets/55259e41-0209-42e3-a01f-103e6d891d71" />

       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       x=new_data[features].values


<img width="296" height="171" alt="499486410-0fcc6e68-4f69-4e3e-a1ef-ff85cf04efe6" src="https://github.com/user-attachments/assets/722eb80c-1ffe-464a-9937-7f9c85aacec1" />

       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split       
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matric
       data=pd.read_csv("income(1) (1).csv")
       prediction=KNN_classifier.predict(test_x)
       confusionMatrix=confusion_matrix(test_y, prediction)

<img width="182" height="58" alt="499486860-87972443-d831-43ff-9375-876d65c7eed0" src="https://github.com/user-attachments/assets/59a156ed-3070-4fe9-a657-4725a991ff3a" />



       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       accuracy_score=accuracy_score(test_y,prediction)
       print(accuracy_score)

<img width="257" height="63" alt="499487283-54d7b47d-c34c-40a7-a696-eaef8918c562" src="https://github.com/user-attachments/assets/b397daec-0c08-4c18-81fd-e4cb9b04c344" />


       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       print("Misclassified Samples : %d" % (test_y !=prediction).sum())

<img width="313" height="40" alt="499487669-224b8a9e-b291-4610-bade-71df5a86363b" src="https://github.com/user-attachments/assets/24bfb49c-6eac-4b55-bfab-02071a7ac23d" />


       import pandas as pd
       import numpy as np
       import seaborn as sns
       from sklearn.model_selection import train_test_split
       from sklearn.neighbors import KNeighborsClassifier
       from sklearn.metrics import accuracy_score, confusion_matrix
       data=pd.read_csv("income(1) (1).csv")
       data.shape
       
<img width="178" height="61" alt="499488168-404d8eba-31ed-4a78-b261-6cccf9d03f84" src="https://github.com/user-attachments/assets/4492b45e-16a2-4ef7-9c8a-a073925b36cb" />

       import pandas as pd
       from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
       data={
       'Feature1': [1,2,3,4,5],
       'Feature2': ['A','B','C','A','B'],
       'Feature3': [0,1,1,0,1],
       'Target' : [0,1,1,0,1]
       }
       df=pd.DataFrame(data)
       x=df[['Feature1','Feature3']]
       y=df[['Target']]
       selector=SelectKBest(score_func=mutual_info_classif,k=1)
       x_new=selector.fit_transform(x,y)
       selected_feature_indices=selector.get_support(indices=True)
       selected_features=x.columns[selected_feature_indices]
       print("Selected Features:")
       print(selected_features)

<img width="1264" height="146" alt="499488623-87cebaa8-4aea-4273-ad15-41f445c2b565" src="https://github.com/user-attachments/assets/300dc7bb-ecf7-4572-89b3-b5b1749d6cfc" />

       import pandas as pd
       import numpy as np
       from scipy.stats import chi2_contingency
       import seaborn as sns
       tips=sns.load_dataset('tips')
       tips.head()

<img width="572" height="214" alt="499489021-3471b870-a540-4660-846a-d9df831a4b8f" src="https://github.com/user-attachments/assets/9f1839d9-eddc-414a-b2db-8f88f6c9b4a6" />
       
       import pandas as pd
       import numpy as np
       from scipy.stats import chi2_contingency
       import seaborn as sns
       tips=sns.load_dataset('tips')
       tips.time.unique()

<img width="601" height="83" alt="499489435-59d79e46-5a0b-4298-880e-aa7d7e226d08" src="https://github.com/user-attachments/assets/d8d5de25-3f22-4133-bd9e-5a3eb0442d3a" />


       import pandas as pd
       import numpy as np
       from scipy.stats import chi2_contingency
       import seaborn as sns
       tips=sns.load_dataset('tips')
       contingency_table=pd.crosstab(tips['sex'],tips['time'])
       print(contingency_table)


<img width="252" height="115" alt="499489955-32cacb3c-fa56-4a8b-82db-59dc30f7dfab" src="https://github.com/user-attachments/assets/b6cf1d3b-406c-4a80-ab7b-f119e08b0917" />

       import pandas as pd
       import numpy as np
       from scipy.stats import chi2_contingency
       import seaborn as sns
       tips=sns.load_dataset('tips')
       chi2,p,,=chi2_contingency(contingency_table)
       print(f"Chi-Square Statistics: {chi2}")
       print(f"P-Value: {p}")

<img width="428" height="63" alt="499490487-86cb52fb-0cfa-4acf-8c5b-71c3c49968f0" src="https://github.com/user-attachments/assets/80c83999-7029-4a48-80ac-92ef2a43d189" />




# RESULT:
Thus feature scaling and feature selection process are performed.

