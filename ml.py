# 1. Data Manipulation and Analysis
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import db

# 3. Data Preprocessing and Feature Scaling
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# 4. Machine Learning Models
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# 5. Model Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


def predict_upcoming_sem(filepath):

    # 1. Train the ML model with the previous dataset (from PostgreSQL database)
    get_all_records = db.get_all_records()

    columns = ['stu_id', 'intake_sem', 'intake_year', 'gender', 'nationality', 'curr_year', 'bm_result', 'pgrm_code', 
               'pgrm_name', 'pgrm_duration', 'pl_id', 'pl_name', 'sbj_code', 'sbj_cat', 'sbj_name', 'sbj_class', 'credit_hours', 
               'sbj_year', 'pre_req', 'sem_id', 'sem_name', 'sem_duration', 'stu_grade', 'sbj_status', 'upcoming_sem']

    df = pd.DataFrame(get_all_records, columns=columns)
    
    # Remove white spaces
    df['stu_grade'] = df['stu_grade'].str.strip().str.upper()

    # Replace the Byte-Order Mark character with an empty string
    df['stu_grade'] = df['stu_grade'].replace('Ï»¿CR1', 'CR1')

    # Replace NaN values in 'stu_grade' with 'Not Attempted'
    df.loc[:, "stu_grade"] = df["stu_grade"].fillna("Not Attempted")

    # Replace NaN values in 'stu_grade' with 'Not Attempted'
    df.loc[:, "upcoming_sem"] = df["upcoming_sem"].fillna("Completed")

    # Replace NaN values with 'Completed'
    df["upcoming_sem"] = df["upcoming_sem"].fillna("Completed")

    # Strip leading/trailing spaces and replace empty strings with 'Completed'
    df["upcoming_sem"] = df["upcoming_sem"].str.strip().replace("", "Completed")

    # Extract the last three characters and convert to integer
    df["stu_id"] = df["stu_id"].str[-3:].astype(int)

    # Encode the intake semester with One Hot Encoder
    ohe_intake_sem = pd.get_dummies(df, columns = ['intake_sem'], dtype='int')
    df = ohe_intake_sem

    # Encode the pgrm_code with One Hot Encoder
    ohe_pgrm_code = pd.get_dummies(df, columns = ['pgrm_code'], dtype='int')
    df = ohe_pgrm_code

    # Encode the pl_id with One Hot Encoder
    ohe_pl_id = pd.get_dummies(df, columns = ['pl_id'], dtype='int')
    df = ohe_pl_id

    # Encode the sbj_code with One Hot Encoder
    ohe_sbj_code = pd.get_dummies(df, columns = ['sbj_code'], dtype = 'int')
    df = ohe_sbj_code

    # Encode the sbj_class with One Hot Encoder
    ohe_sbj_class = pd.get_dummies(df, columns = ['sbj_class'], dtype = 'int')
    df = ohe_sbj_class

    # Encode the pre-requisite subject with One Hot Encoder
    ohe_pre_req = pd.get_dummies(df, columns = ['pre_req'], dtype='int')
    df = ohe_pre_req

    # Encode the semester ID with One Hot Encoder
    ohe_sem_id = pd.get_dummies(df, columns = ['sem_id'], dtype='int')
    df = ohe_sem_id

    # Encode the semester name with One Hot Encoder
    ohe_sem_name = pd.get_dummies(df, columns = ['sem_name'], dtype='int')
    df = ohe_sem_name

    # Encode the student grade with One Hot Encoder
    ohe_stu_grade = pd.get_dummies(df, columns = ['stu_grade'], dtype='int')
    df = ohe_stu_grade

    # Encode the subject status with One Hot Encoder
    ohe_sbj_status = pd.get_dummies(df, columns = ['sbj_status'], dtype='int')
    df = ohe_sbj_status

    # Encode the subject category with One Hot Encoder
    ohe_sbj_cat = pd.get_dummies(df, columns = ['sbj_cat'], dtype='int')
    df = ohe_sbj_cat

    # Perform Label Encoding
    label_encoder = preprocessing.LabelEncoder()
    df['gender'] = label_encoder.fit_transform(df['gender'])
    df['nationality'] = label_encoder.fit_transform(df['nationality'])
    df['bm_result'] = label_encoder.fit_transform(df['bm_result'])

    # Drop Subject Name, Program Name, and PL Name.
    df.drop(['sbj_name', 'pgrm_name', 'pl_name'], axis = 1, inplace = True)

    # Specify the desired order for 'upcoming_sem'
    semester_order = ['AUG-25', 'MAY-26', 'JAN-25', 'Completed', 'MAY-25', 'AUG-27', 'JAN-27',
                    'JAN-26', 'AUG-26', 'MAY-27', 'AUG-28']

    # Create a dictionary mapping each semester to its corresponding index
    semester_mapping = {semester: index for index, semester in enumerate(semester_order)}

    # Apply the custom encoding
    df['upcoming_sem'] = df['upcoming_sem'].map(semester_mapping)

    # Separate the features (X) and the target var (Y)
    X = df.drop('upcoming_sem', axis=1)
    y = df['upcoming_sem']

    # Split the student performance dataset into training and testing sets
    # 20% of the data is allocated for testing, and 80% of the data is allocated for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # --------------------------------------------------1. RF----------------------------------------------------------------
    # Train the Random Forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Calculate the accuracy
    rfAccuracy = accuracy_score(y_test, y_pred)

    # Calculate precision, recall, and F1-score (macro average) with zero_division handling
    rfPrecision = precision_score(y_test, y_pred, average='macro', zero_division=1)  # Prevent undefined precision
    rfRecall = recall_score(y_test, y_pred, average='macro', zero_division=1)  # Prevent undefined recall
    rfF1Score = f1_score(y_test, y_pred, average='macro', zero_division=1)  # Prevent undefined F1-score

    # --------------------------------------------------2. Decision Tree----------------------------------------------------------------
    # Train the Support Vector Machine (SVM) model
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)

    # Calculate the accuracy
    dtAccuracy = accuracy_score(y_test, y_pred)

    # Calculate precision, recall, and F1-score (macro average)
    dtPrecision = precision_score(y_test, y_pred, average='macro', zero_division=1)  # Handle ill-defined precision
    dtRecall = recall_score(y_test, y_pred, average='macro', zero_division=1)  # Handle ill-defined recall
    dtF1Score = f1_score(y_test, y_pred, average='macro', zero_division=1)  # Handle ill-defined F1-score

    # --------------------------------------------------3. KNN----------------------------------------------------------------
    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust K as needed
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Calculate the accuracy
    knnAccuracy = accuracy_score(y_test, y_pred)

    # Calculate precision, recall, and F1-score (macro average)
    knnPrecision = precision_score(y_test, y_pred, average='macro', zero_division=1)  # Handle ill-defined precision
    knnRecall = recall_score(y_test, y_pred, average='macro', zero_division=1)  # Handle ill-defined recall
    knnF1Score = f1_score(y_test, y_pred, average='macro', zero_division=1)  # Handle ill-defined F1-score

    # --------------------------------------------------4. XGB----------------------------------------------------------------
    # Train the XGBoost model
    xgb = XGBClassifier(random_state=42)  # Initialize the XGBClassifier
    xgb.fit(X_train, y_train)  # Fit the model on the training data
    xgb_y_pred = xgb.predict(X_test)  # Predict (without decoding for metrics)

    # Calculate the accuracy
    xgbAccuracy = accuracy_score(y_test, xgb_y_pred)

    # Calculate precision, recall, and F1-score (macro average)
    xgbPrecision = precision_score(y_test, xgb_y_pred, average='macro', zero_division=1)  # Handle ill-defined precision
    xgbRecall = recall_score(y_test, xgb_y_pred, average='macro', zero_division=1)  # Handle ill-defined recall
    xgbF1Score = f1_score(y_test, xgb_y_pred, average='macro', zero_division=1)  # Handle ill-defined F1-score

    # -----------------------------------------------Print the Results------------------------------------------------------------

    # Print RF Results
    print(f"Accuracy: {rfAccuracy:.4f}")
    print(f"Precision: {rfPrecision:.4f}")
    print(f"Recall: {rfRecall:.4f}")
    print(f"F1-score: {rfF1Score:.4f}")
    print()

    # Print SVM results
    print(f"Decision Tree Accuracy: {dtAccuracy:.4f}")
    print(f"Decision Tree Precision: {dtPrecision:.4f}")
    print(f"Decision Tree Recall: {dtRecall:.4f}")
    print(f"Decision Tree F1-score: {dtF1Score:.4f}")
    print()

    # Print KNN results
    print(f"KNN Accuracy: {knnAccuracy:.4f}")
    print(f"KNN Precision: {knnPrecision:.4f}")
    print(f"KNN Recall: {knnRecall:.4f}")
    print(f"KNN F1-score: {knnF1Score:.4f}")
    print()

    # Print XGB results
    print(f"XGBoost Accuracy: {xgbAccuracy:.4f}")
    print(f"XGBoost Precision: {xgbPrecision:.4f}")
    print(f"XGBoost Recall: {xgbRecall:.4f}")
    print(f"XGBoost F1-score: {xgbF1Score:.4f}")

    # Initialize the best accuracy to a very low value
    bestAcc = float('-inf')  # or bestAcc = -1

    # Compare models and update bestAcc and bestModel
    if rfAccuracy > bestAcc:
        bestAcc = rfAccuracy
        bestModel = "Random Forest"
    if dtAccuracy > bestAcc:
        bestAcc = dtAccuracy
        bestModel = "Decision Tree"
    if knnAccuracy > bestAcc:
        bestAcc = knnAccuracy
        bestModel = "KNN"
    if xgbAccuracy > bestAcc:
        bestAcc = xgbAccuracy
        bestModel = "XGB"

    # Print the best model and its accuracy
    print('\nBest Model: ', bestModel)
    print('Accuracy  : ', bestAcc)

    # Perform the prediction on the test set
    y_pred = xgb.predict(X_test)

    # Now calculate the accuracy of the predictions
    # Chosen model is xgb as it has the highest accuracy
    xgbAccuracy = accuracy_score(y_test, y_pred)

    # Print the accuracy
    print(f"\nXGB Accuracy on Test Set: {xgbAccuracy:.4f}")

    # You can also view the predicted values (y_pred) vs actual values (y_test)
    print("\nPredictions vs Actual:")
    for i in range(10):  # Display the first 10 predictions
        # Ensure y_test is a Series to correctly index it
        print(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}")

    # Load a new comma-separated values (csv) file into a DataFrame object
    new_df = pd.read_csv(filepath)

    # Replace NaN values with "Not Attempted"
    new_df['stu_grade'] = new_df['stu_grade'].fillna("NOT ATTEMPTED")

    # Remove leading and trailing spaces
    new_df['stu_grade'] = new_df['stu_grade'].str.strip()

    # Extract the last three characters and convert to integer
    new_df["stu_id"] = new_df["stu_id"].str[-3:].astype(int)

    # Encode the intake semester with One Hot Encoder
    ohe_intake_sem = pd.get_dummies(new_df, columns = ['intake_sem'], dtype='int')
    new_df = ohe_intake_sem

    # Encode the pgrm_code with One Hot Encoder
    ohe_pgrm_code = pd.get_dummies(new_df, columns = ['pgrm_code'], dtype='int')
    new_df = ohe_pgrm_code

    # Encode the pl_id with One Hot Encoder
    ohe_pl_id = pd.get_dummies(new_df, columns = ['pl_id'], dtype='int')
    new_df = ohe_pl_id

    # Encode the sbj_code with One Hot Encoder
    ohe_sbj_code = pd.get_dummies(new_df, columns = ['sbj_code'], dtype = 'int')
    new_df = ohe_sbj_code

    # Encode the sbj_class with One Hot Encoder
    ohe_sbj_class = pd.get_dummies(new_df, columns = ['sbj_class'], dtype = 'int')
    new_df = ohe_sbj_class

    # Encode the pre-requisite subject with One Hot Encoder
    ohe_pre_req = pd.get_dummies(new_df, columns = ['pre_req'], dtype='int')
    new_df = ohe_pre_req

    # Encode the semester ID with One Hot Encoder
    ohe_sem_id = pd.get_dummies(new_df, columns = ['sem_id'], dtype='int')
    new_df = ohe_sem_id

    # Encode the semester name with One Hot Encoder
    ohe_sem_name = pd.get_dummies(new_df, columns = ['sem_name'], dtype='int')
    new_df = ohe_sem_name

    # Encode the student grade with One Hot Encoder
    ohe_stu_grade = pd.get_dummies(new_df, columns = ['stu_grade'], dtype='int')
    new_df = ohe_stu_grade

    # Encode the subject status with One Hot Encoder
    ohe_sbj_status = pd.get_dummies(new_df, columns = ['sbj_status'], dtype='int')
    new_df = ohe_sbj_status

    # Encode the subject category with One Hot Encoder
    ohe_sbj_cat = pd.get_dummies(new_df, columns = ['sbj_cat'], dtype='int')
    new_df = ohe_sbj_cat

    # Perform Label Encoding
    label_encoder = preprocessing.LabelEncoder()
    new_df['gender'] = label_encoder.fit_transform(new_df['gender'])
    new_df['nationality'] = label_encoder.fit_transform(new_df['nationality'])
    new_df['bm_result'] = label_encoder.fit_transform(new_df['bm_result'])

    # Drop Subject Name, Program Name, and PL Name.
    new_df.drop(['sbj_name', 'pgrm_name', 'pl_name'], axis = 1, inplace = True)

    X_new = new_df.drop(columns=['upcoming_sem'])

    # Get the columns in the same order as the trained model
    X_new = X_new[X_train.columns]

    # Make predictions
    predicted_upcoming_sem = xgb.predict(X_new)
    new_df['predicted_upcoming_sem'] = predicted_upcoming_sem
    new_df.head()

    # Define the mapping between encoded values and original categories
    decode_mapping = {
        0: "2025-08-01",
        1: "2026-05-01",
        2: "2025-01-01",
        3: None,
        4: "2025-05-01",
        5: "2027-08-01",
        6: "2027-01-01",
        7: "2026-01-01",
        8: "2026-08-01",
        9: "2027-05-01",
        10: "2028-08-01"
    }

    # Decode the 'upcoming_sem' column
    new_df['upcoming_sem'] = new_df['predicted_upcoming_sem'].map(decode_mapping)

    # Decode Student ID
    new_df["stu_id"] = "B" + new_df["stu_id"].astype(str).str.zfill(6)

    # Decode intake_sem
    new_df['intake_sem'] = new_df[['intake_sem_AUG', 'intake_sem_JAN', 'intake_sem_MAY']].idxmax(axis=1)
    new_df['intake_sem'] = new_df['intake_sem'].str.replace('intake_sem_', '')  # Remove prefix
    new_df.drop(['intake_sem_AUG', 'intake_sem_JAN', 'intake_sem_MAY'], axis=1, inplace=True)

    # Decode pgrm_code
    new_df['pgrm_code'] = new_df[[col for col in new_df.columns if col.startswith('pgrm_code_')]].idxmax(axis=1)
    new_df['pgrm_code'] = new_df['pgrm_code'].str.replace('pgrm_code_', '')
    new_df.drop([col for col in new_df.columns if col.startswith('pgrm_code_')], axis=1, inplace=True)

    # Decode pl_id
    new_df['pl_id'] = new_df[[col for col in new_df.columns if col.startswith('pl_id_')]].idxmax(axis=1)
    new_df['pl_id'] = new_df['pl_id'].str.replace('pl_id_', '')
    new_df.drop([col for col in new_df.columns if col.startswith('pl_id_')], axis=1, inplace=True)

    # Decode sbj_code
    new_df['sbj_code'] = new_df[[col for col in new_df.columns if col.startswith('sbj_code_')]].idxmax(axis=1)
    new_df['sbj_code'] = new_df['sbj_code'].str.replace('sbj_code_', '')
    new_df.drop([col for col in new_df.columns if col.startswith('sbj_code_')], axis=1, inplace=True)

    # Decode sbj_class
    new_df['sbj_class'] = new_df[[col for col in new_df.columns if col.startswith('sbj_class_')]].idxmax(axis=1)
    new_df['sbj_class'] = new_df['sbj_class'].str.replace('sbj_class_', '')
    new_df.drop([col for col in new_df.columns if col.startswith('sbj_class_')], axis=1, inplace=True)

    # Decode pre_req
    new_df['pre_req'] = new_df[[col for col in new_df.columns if col.startswith('pre_req_')]].idxmax(axis=1)
    new_df['pre_req'] = new_df['pre_req'].str.replace('pre_req_', '')
    new_df.drop([col for col in new_df.columns if col.startswith('pre_req_')], axis=1, inplace=True)

    # Decode sem_id
    new_df['sem_id'] = new_df[[col for col in new_df.columns if col.startswith('sem_id_')]].idxmax(axis=1)
    new_df['sem_id'] = new_df['sem_id'].str.replace('sem_id_', '')
    new_df.drop([col for col in new_df.columns if col.startswith('sem_id_')], axis=1, inplace=True)

    # Decode sem_name
    new_df['sem_name'] = new_df[[col for col in new_df.columns if col.startswith('sem_name_')]].idxmax(axis=1)
    new_df['sem_name'] = new_df['sem_name'].str.replace('sem_name_', '')
    new_df.drop([col for col in new_df.columns if col.startswith('sem_name_')], axis=1, inplace=True)

    # Decode stu_grade
    new_df['stu_grade'] = new_df[[col for col in new_df.columns if col.startswith('stu_grade_')]].idxmax(axis=1)
    new_df['stu_grade'] = new_df['stu_grade'].str.replace('stu_grade_', '')
    new_df.drop([col for col in new_df.columns if col.startswith('stu_grade_')], axis=1, inplace=True)

    # Decode sbj_status
    new_df['sbj_status'] = new_df[[col for col in new_df.columns if col.startswith('sbj_status_')]].idxmax(axis=1)
    new_df['sbj_status'] = new_df['sbj_status'].str.replace('sbj_status_', '')
    new_df.drop([col for col in new_df.columns if col.startswith('sbj_status_')], axis=1, inplace=True)

    # Decode sbj_cat
    new_df['sbj_cat'] = new_df[[col for col in new_df.columns if col.startswith('sbj_cat_')]].idxmax(axis=1)
    new_df['sbj_cat'] = new_df['sbj_cat'].str.replace('sbj_cat_', '')
    new_df.drop([col for col in new_df.columns if col.startswith('sbj_cat_')], axis=1, inplace=True)

    # Recreate the LabelEncoder for each column if needed
    gender_decoder = preprocessing.LabelEncoder()
    gender_decoder.fit(['M', 'F'])  # Provide the original categories
    new_df['gender'] = gender_decoder.inverse_transform(new_df['gender'])

    nationality_decoder = preprocessing.LabelEncoder()
    nationality_decoder.fit(['MALAYSIAN', 'INTERNATIONAL'])  # Provide original categories
    new_df['nationality'] = nationality_decoder.inverse_transform(new_df['nationality'])

    bm_result_decoder = preprocessing.LabelEncoder()
    bm_result_decoder.fit(['CREDIT', 'FAIL'])  # Provide original categories
    new_df['bm_result'] = bm_result_decoder.inverse_transform(new_df['bm_result'])

    # Drop predicted_upcoming_sem
    new_df = new_df.drop(columns=['predicted_upcoming_sem'])

    new_df['stu_grade'] = new_df['stu_grade'].replace('NOT ATTEMPTED', 'Not Attempted')

    # Apply the function to each row
    new_df["supposed_to_be_taken"] = new_df.apply(
        lambda row: calculate_supposed_to_be_taken(row["intake_year"], row["sbj_year"], row["sem_name"]), axis=1
    )

    new_df.to_csv('uploads/help.csv', index=False)
    return new_df

def calculate_supposed_to_be_taken(intake_year, sbj_year, sem_name):
    # Define semester month mapping
    sem_months = {'JAN': 1, 'MAY': 5, 'AUG': 8}
    
    # Calculate the year based on sbj_year
    year = intake_year + (sbj_year - 1)
    
    # Get the month from sem_name
    month = sem_months.get(sem_name, 1)  # Default to 1 (January) if sem_name is invalid
    
    # Construct the date
    return f"{year}-{month:02d}-01"
