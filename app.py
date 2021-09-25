from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("AirQualityUCI.csv")

l=list(df["CO_level"].unique())
for num,var in enumerate(l):
    num+=1
    df["CO_level"].replace(var, num, inplace=True)

da=[]
for var in df["Date"]:
    try:
        temp = var.split("/")
        da.append(int(temp[1]))
    except:
        da.append(np.nan)

df1=df.join(pd.DataFrame({"date":da}))
df1.head()

da1=[]
for var in df["Time"]:
    try:
        temp = var.split(":")
        temp1=(int(temp[0])+int(temp[1])+int(temp[2]))/3
        da1.append(temp1)
    except:
        da1.append(np.nan)

df2=df1.join(pd.DataFrame({"time":da1}))
df2.head()

df3 = df2.drop(["Date","Time"], axis=1)
df3.head()

X = df3.drop("CO_level", axis=1)
y = df3["CO_level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

sc=StandardScaler()
sc.fit(X_train,y_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


model = joblib.load("air_co_level_prediction_model.pkl")


def air_co_level_prediction(model, CO_GT, PT08_S1_CO, NMHC_GT, C6H6_GT, PT08_S2_NMHC, Nox_GT,
       PT08_S3_Nox, NO2_GT, PT08_S4_NO2, PT08_S5_O3, T, RH, AH, date, time):
#     for num,var in enumerate(df_value):
#         if var == Decision:
#             Decision = num
            
    x = np.zeros(len(X.columns))
    x[0] = CO_GT
    x[1] = PT08_S1_CO
    x[2] = NMHC_GT
    x[3] = C6H6_GT
    x[4] = PT08_S2_NMHC
    x[5] = Nox_GT
    x[6] = PT08_S3_Nox
    x[7] = NO2_GT
    x[8] = PT08_S4_NO2
    x[9] = PT08_S5_O3
    x[10] = T
    x[11] = RH
    x[12] = AH
    x[13] = date
    x[14] = time
    
    x = sc.transform([x])[0]
    return model.predict([x])[0]

app=Flask(__name__)

@app.route("/")
def home():
    # value7 = list(df2["Name"].value_counts().keys())
    # value7.sort()
    # value10 = list(df3["City"].value_counts().keys())
    # value10.sort()
    # value11 = list(df4["State"].value_counts().keys())
    # value11.sort()
    return render_template("index.html")    #,value=value7, value01=value10,value02=value11

# CO_GT, PT08_S1_CO, NMHC_GT, C6H6_GT, PT08_S2_NMHC, Nox_GT,
#        PT08_S3_Nox, NO2_GT, PT08_S4_NO2, PT08_S5_O3, T, RH, AH, date, time


@app.route("/predict", methods=["POST"])
def predict():
    CO_GT = request.form["CO_GT"]
    PT08_S1_CO = request.form["PT08_S1_CO"]
    NMHC_GT = request.form["NMHC_GT"]
    C6H6_GT = request.form["C6H6_GT"]
    PT08_S2_NMHC = request.form["PT08_S2_NMHC"]
    Nox_GT = request.form["Nox_GT"]
    PT08_S3_Nox = request.form["PT08_S3_Nox"]
    NO2_GT = request.form["NO2_GT"]
    PT08_S4_NO2 = request.form["PT08_S4_NO2"]
    PT08_S5_O3 = request.form["PT08_S5_O3"]
    T = request.form["T"]
    RH = request.form["RH"]
    AH = request.form["AH"]
    date = request.form["date"]
    time = request.form["time"]
    
    predicated_price = air_co_level_prediction(model, CO_GT, PT08_S1_CO, NMHC_GT, C6H6_GT, PT08_S2_NMHC, Nox_GT,
                                                PT08_S3_Nox, NO2_GT, PT08_S4_NO2, PT08_S5_O3, T, RH, AH, date, time)

    if predicated_price==0:
        return render_template("index.html", prediction_text="CO_level of:- {}".format(predicated_price))
    elif predicated_price==1:
        return render_template("index.html", prediction_text="CO_level of:- {}".format(predicated_price))
    elif predicated_price==2:
        return render_template("index.html", prediction_text="CO_level of:- {}".format(predicated_price))
    elif predicated_price==3:
        return render_template("index.html", prediction_text="CO_level of:- {}".format(predicated_price))
    else:
        return render_template("index.html", prediction_text="CO_level of:- {}".format(predicated_price))


if __name__ == "__main__":
    app.run()    
    