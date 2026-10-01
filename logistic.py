from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
import pandas as pd 
import joblib

df = pd.read_csv(r'C:\Users\Admin\Desktop\3skill\dataset_12000_records.csv')

df = df.drop(columns=["Patient_ID"])

X = df.drop(columns = ["Readmitted_30_Days"])
y = df['Readmitted_30_Days']

X = pd.get_dummies(X, drop_first = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(
    penalty='elasticnet',            
    C=1.0,                   
    solver='saga',          
    max_iter=1000,           
    class_weight='balanced', 
    random_state=42          
)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print ("Predicted values:", y_pred)

cm = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy_score(y_test, y_pred)*100}')
print(f'Confusion Matrix:\n{cm}')



joblib.dump(model, 'logistic_model.pkl')


loaded_model = joblib.load('logistic_model.pkl')
prediction = loaded_model.predict(X_test)



'''
# Get probability predictions
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Calculate AUC
roc_auc = auc(fpr, tpr)
print(f'ROC-AUC Score: {roc_auc}')
'''
