from api42wrapper.api42 import Api42
from decouple import config
from pprint import pprint
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,roc_curve
import argparse

client = Api42(
	uid=config('INTRA_API_V2_UID'),
	secret=config('INTRA_API_V2_SECRET'),
	scope='public',
	uidv3=config('INTRA_API_V3_UID'),
	secretv3=config('INTRA_API_V3_SECRET'),
	username=config('INTRA_API_V3_USERNAME'),
	password=config('INTRA_API_V3_PASSWORD'),
	totp=config('INTRA_API_V3_TOTP')
)


# /v2/users/{uid}/scale_teams - fetching records of individual as corrector
def get_users(client: Api42,cursus_id:int = 21,primary_campus_id:int = 14):
    status, data = client.get(f"/v2/cursus/{cursus_id}/users",fetch_all=True,
                              filter={'primary_campus_id':primary_campus_id})
    if status != 200:
        return status
    return pd.DataFrame(data)

def get_odds_ratios(model,feature_names):
    """
    calculate odds ratio for each feature
    """
    coefficients = model.coef_[0]
    odds_ratios = np.exp(coefficients)

    print("COEFFICIENTS AND ODDS RATIOS")
    print('='*40)
    for i, (coef, odds_ratio,feature) in enumerate(zip(coefficients, odds_ratios,feature_names)):
        print(f"{feature}:")
        print(f"{coef}:{coef:.4f}")
        print(f"{odds_ratio}:{odds_ratio:.4f}")
        print(f"Interpretation: 1-unit increase in {feature} multiplies odds by {odds_ratio:.4f}")

    return coefficients,odds_ratios

# 6. Prediction Function
def predict_dropout_probability(attendance):
    """
    Predict dropout probability based on attendance
    """
    prob = model.predict_proba([[attendance]])[0][1]
    return prob

if __name__== "__main__":
    parser = argparse.ArgumentParser( 
        description="AAAA"
    )
    parser.add_argument('-p','--prepare',action='store_true',help="prepare for the dataframe")
    parser.add_argument('-m','--no_model',action='store_true',help="do not build model")
    args = parser.parse_args()
    
    if args.prepare:
        aggregated = pd.read_csv("AllCohortsOverview_aggregated.csv")
        attendance = pd.read_csv("attendance_prediction_model/cc_attendance_cleaned.csv")[['user_id','cc_avg_login_hours']]
        users = get_users(client)[['id','login']].rename(columns={'id':'user_id'})
        aggregated = pd.merge(users,aggregated,how = 'right',on='login')
        df = pd.merge(aggregated,attendance,how = 'left',on = 'user_id')
        df.to_csv('AllCohortsOverview_aggregated_2.csv')

    if not args.no_model:
        df = pd.read_csv("AllCohortsOverview_aggregated_2.csv")
        df = df[(df['cc_avg_login_hours']!= 0)&(df['cc_avg_login_hours'].notna())]
        print(df.info)
       

        data_exploration = True
        visual = True
        logistic = True

        if data_exploration:
            print("\n=================DATA EXPLORATION==================")
            print(df.groupby('dropout')['cc_avg_login_hours'].describe())

        if visual:
            fig, axes = plt.subplots(1,2,figsize = (12,5))
            # Box plot - see distribution differences
            df.boxplot(column='cc_avg_login_hours',by='dropout',ax=axes[0])
            axes[0].set_title('Attendance Distribution by Dropout Status')
            axes[0].set_xlabel('Dropout')

            # Scatter plot - see relationship
            axes[1].scatter(df['cc_avg_login_hours'],df['dropout'],alpha = 0.6)
            axes[1].set_xlabel('Attendance')
            axes[1].set_ylabel('dropout')
            axes[1].set_title('Attendance vs. Dropout')

            plt.tight_layout()
            plt.show()

        if logistic:
            # 3. Logistic Regression Model
            print("==================Logistic Regression Model============")
            X = df[['cc_avg_login_hours']]
            y = df['dropout']
            # split data
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=20, stratify=y)
            # train model
            model = LogisticRegression(random_state=42)
            model.fit(X_train,y_train)
            # 3.4 Model coefficients
            print(f"Intercept: {model.intercept_[0]:.4f}")
            print(f"Attendance coefficients: {model.coef_[0][0]:.4f}")

            
            # Calculate odds ratio
            odds_ratio_attendance = np.exp(model.coef_[0][0])
            print(f"Odds Ratio: {odds_ratio_attendance:.4f}")
            print(f"Interpretation: For each 1-unit increase in attendance, odds of dropout multiplies by {odds_ratio_attendance:.4f}")

            # 4. Model Evaluation
            print("\n=== Model Evaluation ===")

            # 4.1 Training predictions
            y_train_pred = model.predict(X_train)
            y_train_proba = model.predict_proba(X_train)[:, 1]

            # 4.2 Test predictions
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]
            fpr,tpr,thresholds = roc_curve(y_test,y_test_proba)
            # Find optimal threshold (Youden's J statistics)           
            optimal_idx = np.argmax(tpr-fpr) # find the index of that threshold with which true positive rate has the biggest difference with false positive rate
            optimal_threshold = thresholds[optimal_idx]
            attendance_threshold_test= (np.log(1/optimal_threshold-1)+model.intercept_[0])/(-model.coef_[0][0])
            print(f"Optimal threshold (test): {optimal_threshold:.3f}")
            print(f"Attendance threshold (test): {attendance_threshold_test:.3f}")
            y_pred_custom = (y_test_proba >= optimal_threshold).astype(int) # predicted labels are customed to the adjusted threshold ( Before that the default threshold is 0.5)

            # 4.3 Evaluation metrics

            print("\nTest performance:")
            print(classification_report(y_test, y_pred_custom))
            print(f"AUC: {roc_auc_score(y_test, y_pred_custom):.4f}")

            # 4.4 Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            print(f"\n5-fold Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            # 5. Visualization of Results
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # 5.1 ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            axes[0].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_test_proba):.4f})')
            axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[0].set_xlabel('False Positive Rate')
            axes[0].set_ylabel('True Positive Rate')
            axes[0].set_title('ROC Curve')
            axes[0].legend()

            # 5.2 Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_custom)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Actual')
            axes[1].set_title('Confusion Matrix')

            # 5.3 Predicted Probability Distribution
            axes[2].hist(y_test_proba[y_test==0], alpha=0.7, label='No Dropout', bins=20)
            axes[2].hist(y_test_proba[y_test==1], alpha=0.7, label='Dropout', bins=20)
            axes[2].set_xlabel('Predicted Probability')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title('Predicted Probability Distribution')
            axes[2].legend()

            plt.tight_layout()
            plt.savefig('Model Performance.png')
            plt.show()

            # Example predictions
            print("\n=== Example Predictions ===")
            test_attendance = [6, 8, 10, 17, 24]
            for att in test_attendance:
                prob = predict_dropout_probability(att)
                print(f"Attendance = {att} hours, Dropout probability = {prob:.4f}")

            # 7. Model Diagnostics
            print("\n=== Model Diagnostics ===")

            # 7.1 Check linearity assumption
            attendance_range = np.linspace(df['cc_avg_login_hours'].min(), df['cc_avg_login_hours'].max(), 100)
            probs = pd.Series(attendance_range).apply(predict_dropout_probability)
            # log_odds = model.intercept_[0] + model.coef_[0][0] * attendance_range

            plt.figure(figsize=(10, 6))
            plt.plot(attendance_range, probs)
            plt.xlabel('weekly login hours')
            plt.ylabel('dropout probability')
            plt.title('Dropout probability vs. weekly login hours')
            plt.grid(True)
            plt.savefig("DropoutAttendance.png")
            plt.show()
            

            # 7.2 Residual analysis
            residuals = y_test - y_test_proba
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test_proba, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Probability')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.grid(True)
            plt.savefig("residuals.png")
            plt.show()
            

            print("\n=== Model Summary ===")
            print(f"1. Attendance is a {'significant' if abs(model.coef_[0][0]) > 0.05 else 'non-significant'} predictor of dropout")
            print(f"2. Model AUC = {roc_auc_score(y_test, y_test_proba):.4f}, {'good' if roc_auc_score(y_test, y_test_proba) > 0.7 else 'moderate'} performance")
            print(f"3. Next step: Add other predictors to improve model performance")
