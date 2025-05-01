import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, root_mean_squared_error, accuracy_score

# Data Preprocessing
df = pd.read_csv('CA_Weather_Fire_Dataset_1984-2025.csv')
df['DATE_TIME'] = pd.to_datetime(df['DATE'])
df.dropna(inplace=True)
df = df[df['YEAR'] != 2025]

# Define features and target variable 
features = ['PRECIPITATION', 'MAX_TEMP', 'MIN_TEMP', 'AVG_WIND_SPEED',
            'TEMP_RANGE', 'WIND_TEMP_RATIO', 'LAGGED_PRECIPITATION',
            'LAGGED_AVG_WIND_SPEED', 'MONTH']
X = df[features]
y = df['FIRE_START_DAY']

# Split the data into training and testing sets by 1984-2015 and 2016-2024
df = df.sort_values(by='YEAR')  # Replace 'YEAR' with your actual time column
cutoff_year = 2016

train = df[df['YEAR'] < cutoff_year]
test = df[df['YEAR'] >= cutoff_year]

X_train = train[features]
y_train = train['FIRE_START_DAY']

X_test = test[features]
y_test = test['FIRE_START_DAY']


# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)



# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Feature Importance for RF
rf_feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index=X_train.columns,
                                   columns=['Importance']).sort_values('Importance', ascending=False)


# XGBoost Classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Feature Importance for XGBoost
xgb_feature_importances = pd.DataFrame(xgb.feature_importances_,
                                   index=X_train.columns,
                                   columns=['Importance']).sort_values('Importance', ascending=False)





if __name__ == "__main__":
    #Fire Per Year
    # plt.figure(figsize=(10, 4))
    # sns.barplot(x='YEAR', y='FIRE_START_DAY', hue="SEASON", data=df, estimator=sum)
    # plt.title("Fire Starts per Year")
    # plt.ylabel("Number of Fires")
    # plt.xlabel("Year")
    # plt.tight_layout()
    # plt.show()

    # #Fire by Season
    # sns.barplot(x='SEASON', y='FIRE_START_DAY', data=df, estimator=sum)
    # plt.title("Fires by Season")
    # plt.ylabel("Total Fire Starts")
    # plt.show()

    # # Temperature 
    # sns.boxplot(x='FIRE_START_DAY', y='MAX_TEMP', data=df)
    # plt.title("Max Temperature on Fire vs Non-Fire Days")
    # plt.show()

    # # Wind Speed
    # sns.boxplot(x='FIRE_START_DAY', y='AVG_WIND_SPEED', data=df)
    # plt.title("Wind Speed on Fire vs Non-Fire Days")
    # plt.show()

    # # Precipitation
    # sns.boxplot(x='FIRE_START_DAY', y='PRECIPITATION', data=df)
    # plt.title("Precipitation on Fire vs Non-Fire Days")
    # plt.show()


    #Logisitic Regression
    print("Logistic Regression Classification Report")
    print(y_pred_log_reg)
    print(classification_report(y_test, y_pred_log_reg))
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy


    # # Random Forest
    # print("Random Forest Classification Report")
    # print(classification_report(y_test, y_pred_rf))
    # print("Random Forest Feature Importances")
    # print(rf_feature_importances)

    # # XGBoost
    # print("XGBoost Classification Report")
    # print(classification_report(y_test, y_pred_xgb))
    # print("XGBoost Feature Importances")
    # print(xgb_feature_importances)

    # #Graph Feature Importance for RF and XGB
    # plt.figure(figsize=(10, 6))
    # plt.subplot(1, 2, 1) 
    # sns.barplot(x=rf_feature_importances['Importance'], y=rf_feature_importances.index)
    # plt.title("Random Forest Feature Importances")
    # plt.subplot(1, 2, 2)
    # sns.barplot(x=xgb_feature_importances['Importance'], y=xgb_feature_importances.index)
    # plt.title("XGBoost Feature Importances")
    # plt.tight_layout()
    # plt.show()
