import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_theme(style="ticks",font_scale=1.5)

"""
-we have (2448, 7) data -> after cleaning (2442, 7) -> long after (2427, 7)
-changing column names to english
-remove rooms>30, price 2e9, age above 80
-price max: 33M |  avg: 3.8M  | min:0.6M

-Need to encode object data types -> Floor, Location, District
-using xgb and doing kfold cross val
-I assume the performance of the model good enough, we need more data probably
    *R2 score is 0.71, MAPE score is 0.21
"""

df=pd.read_excel("Ankara_house_prices.xlsx")
df.drop(df.columns[0],inplace=True,axis=1)

#SOME DATA CLEANING
df=df.set_axis(["Square (m2)","Rooms","Age","Floor","Location","District","Price (TL)"],axis=1)
df.loc[df["Rooms"]=="Stüdyo"]=1
df["Rooms"]=df["Rooms"].astype("int64")
df=df[(df["Rooms"]<30) & (df["Age"]<80) \
      & (df["Price (TL)"]<=2e7) & (df["Price (TL)"]>1e5)]


def encode(col):
    encoder=LabelEncoder()
    df[col]=encoder.fit_transform(df[[col]])
    
    return encoder

def gridsearch(model):
    parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
    }
    
    grid_search=GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring='r2',
        n_jobs=10,
        cv=5,
        verbose=True
        )
    
    grid_search.fit(X,y)
    
    print(grid_search.best_params_)
    return grid_search.best_estimator_

def train_model(plot=False):
    model=XGBRegressor(n_estimators=200)
    model_best=gridsearch(model)

    model_best.fit(X_train,y_train)
    y_pred=model_best.predict(X_test)

    #kfold
    kfold=KFold(n_splits=5) 
    r2=np.mean(cross_val_score(model,X,y,cv=kfold))
    mape=np.mean(cross_val_score(model,X,y,cv=kfold,scoring='neg_mean_absolute_percentage_error')*-1)

    if plot:
        plt.scatter(y_test,y_pred,color="royalblue",linewidths=0.8,edgecolors="black")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.title(f"R2: {r2:0.2f} | MAPE: {mape:0.2f}")
        plt.axline((0,0),slope=1,linestyle="--",color="gray")

    print("R2 score is %0.2f, MAPE score is %0.2f" %(r2,mape))
    return model_best

# PREPROCESS for ML
enc_floor=encode("Floor")
enc_loc=encode("Location")
enc_dist=encode("District")

#features and the label
X=df.drop("Price (TL)",inplace=False,axis=1)
y=df["Price (TL)"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)

#the model
model=train_model()

#MAKING A PREDICTION of the house im living in
given=np.array([90,3,20,'2. Kat ','Eryaman','Etimesgut']).reshape(1,-1)
df_in=pd.DataFrame(given,columns=['Square (m2)', 'Rooms', 'Age', 'Floor', 'Location', 'District'])

df_in["Floor"]=enc_floor.transform(df_in[["Floor"]])
df_in["Location"]=enc_loc.transform(df_in[["Location"]])
df_in["District"]=enc_dist.transform(df_in[["District"]])
df_in=df_in.astype("float64")

prices=[]
for _ in range(10):
    predicted_price=model.predict(df_in)
    prices.append(predicted_price)
    print("THE PREDICTED PRICE IS ---> %0.2f TL" %predicted_price)

print("\nTHE FINAL PRICE IS ---> %0.2f TL" %np.mean(prices))

"""
WELL, THE ANSWERS ARE CONSISTENT SO THAT IS SOMETHING
I AM CONCLUDING THIS WITH THE FINAL ANSWER OF 2.1M TL
LET ME SEE THE SIMILAR ONES SO WHETHER THE ANSWER MAKES SENSE OR NOT
IT KINDA MAKES SENSE, BUT THE DATA FOR THIS LOCATION IS VERY SCARCE
"""