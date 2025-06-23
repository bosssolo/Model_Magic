# -------------------- Importing All Library at once -----------------# 
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import r2_score, accuracy_score,root_mean_squared_error
import joblib 







st.title("The ML Models Train by Siddhant")

# --------------- Uploading Section ----------------- # 
uploading_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploading_file:
    df = pd.read_csv(uploading_file)
    st.write("Original Data Preview")
    st.dataframe(df.head())

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
        else:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)

    # Label Encoding for categorical columns
    label_encoders = {}
    label_mappings = {}

    for col in df.columns:
        if df[col].dtype == 'object' :
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    st.write("### Cleaned Data Preview")
    st.dataframe(df.head())

    # User selects features and target
    columns = df.columns.tolist()
    X_cols = st.multiselect("Select feature columns (X)", options=columns)
    y_col = st.selectbox(" Select target column (Y)", options=columns)

    model_type = st.selectbox("Choose a Model Type", ["Linear Regression", "Logistic Regression","Support Vector Machine","Decision Tree","KNN","Random Forest"])

    if X_cols and y_col:
        X = df[X_cols]
        y = df[y_col]

        y_encoder = None
        if y_col in label_encoders:
            y_encoder = label_encoders[y_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standard scaling only for Logistic Regression
        if model_type == "Logistic Regression":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if model_type == "Decision Tree":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        if model_type == "Random Forest":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)    
            

        # Model selection
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Support Vector Machine":
            model = SVC()
        elif model_type == "Decision Tree" :
                model = DecisionTreeClassifier()
                # In Decision Tree i have used Hyper Parameter Tunning 
                if len(X_train) > 5:
                    try_paremeter = {
                        "criterion": ['gini', 'entropy'],
                        "max_depth": [5, 10, 15, None],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    }

                    grid = GridSearchCV(model, param_grid=try_paremeter, cv=2, scoring='accuracy')
                    grid.fit(X_train, y_train)

                    model = grid.best_estimator_
                    

            
        elif model_type == "Random Forest" : 
            model = RandomForestClassifier()

                        
        elif model_type == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        
        else:
            model = LogisticRegression()
     
        
        model.fit(X_train, y_train)
        

        y_pred = model.predict(X_test)
        

        st.write("### Model Performance")

        
        if model_type == "Linear Regression":
            score = r2_score(y_test, y_pred)
            score2 = root_mean_squared_error(y_test,y_pred)
            st.write(f"R- score: %",score * 100 )
            st.write(f"RMSE Score: %", score2 * 100 )
            
        elif model_type == "Support Vector Machine":
            accu = accuracy_score(y_test,y_pred)
            st.write(f"Accuracy: %",accu * 100 )


        elif model_type == "Decision Tree":
            acc = accuracy_score(y_test,y_pred)
            
            st.write(f"Accuracy: %",acc * 100 )
            



        elif model_type == "Random Forest":
            
            rop = accuracy_score(y_test,y_pred)
            st.write(f"Accuracy: %",rop * 100 )

  
        elif model_type == "KNN":
            ac = accuracy_score(y_test,y_pred)
            
            st.write(f"Accuracy: %",ac * 100 )

            



        else:
            if y_encoder is not None:
                y_test_decoded = y_encoder.inverse_transform(y_test.astype(int))
                y_pred_decoded = y_encoder.inverse_transform(y_pred.astype(int))
                acc = accuracy_score(y_test_decoded, y_pred_decoded)
                st.write(f"Accuracy: {acc:.1f}")
            else:
                acc = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {acc:.1f}")


    

        # Prediction section
        st.write("### Try a Prediction")

        input_data = {}

        for col in X_cols:
            if col in label_encoders:
                options = list(label_mappings[col].keys())
                selected = st.selectbox(f"Choose value for {col}", options)
                input_data[col] = label_mappings[col][selected]
            else:
                mean_value = df[col].mean()
                if pd.api.types.is_integer_dtype(df[col]):
                    input_val = st.number_input(f"Enter value for {col}", value=int(mean_value), step=1, format="%d")
                else:
                    input_val = st.number_input(f"Enter value for {col}", value=round(mean_value, 2))
                input_data[col] = input_val

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            if model_type == "Logistic Regression":
                input_df = scaler.transform(input_df)

            result = model.predict(input_df)
            


            st.write("### Prediction Result")
            if model_type == "Linear Regression":
                label = y_encoder.inverse_transform([int(result[0])])[0]
                if float(result[0]).is_integer():
                    

                    st.success(f"Prediction: {label}")
                else:
                    st.success(f"Prediction: {label}")
            else:
                if y_encoder is not None:
                    label = y_encoder.inverse_transform([int(result[0])])[0]
                    st.success(f"Prediction: {label}")
                else:
                    if result[0] == 1:
                        st.success("Prediction: Yes")
                    else:
                        st.success("Prediction: No")
        

            

