import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Dự Đoán Chất Lượng Rượu Vang", page_icon=":wine_glass:")

# Tải dữ liệu
def load_data():
    data = {
        'fixed acidity': [7.4, 7.8, 7.8, 11.2, 7.4],
        'volatile acidity': [0.7, 0.88, 0.76, 0.28, 0.7],
        'citric acid': [0.0, 0.0, 0.04, 0.56, 0.0],
        'residual sugar': [1.9, 2.6, 2.3, 1.9, 1.9],
        'chlorides': [0.076, 0.098, 0.092, 0.075, 0.076],
        'free sulfur dioxide': [11.0, 25.0, 15.0, 17.0, 11.0],
        'total sulfur dioxide': [34.0, 67.0, 54.0, 60.0, 34.0],
        'density': [0.9978, 0.9968, 0.9970, 0.9980, 0.9978],
        'pH': [3.51, 3.20, 3.26, 3.16, 3.51],
        'sulphates': [0.56, 0.68, 0.65, 0.58, 0.56],
        'alcohol': [9.4, 9.8, 9.8, 9.8, 9.4],
        'quality': [5, 5, 5, 6, 5]
    }
    return pd.DataFrame(data)

# Load dữ liệu
data = load_data()

# Tạo giao diện Streamlit
st.title("Dự Đoán Chất Lượng Rượu Vang")

# Hiển thị dữ liệu mẫu
st.subheader("Dữ liệu mẫu")
st.write(data)

# Tạo mô hình dự đoán
st.subheader("Dự đoán chất lượng")
model_choice = st.selectbox("Chọn mô hình dự đoán", ["Linear Regression", "Lasso Regression", "Artificial Neural Network", "Stacking"])

# Tách dữ liệu thành train/test
X = data.drop("quality", axis=1)
y = data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện mô hình dựa trên lựa chọn
if model_choice == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

elif model_choice == "Lasso Regression":
    lasso = Lasso()
    parameters = {"alpha": [0.1, 0.5, 1.0]}
    model = GridSearchCV(lasso, parameters, scoring="neg_mean_squared_error", cv=2)  # Giảm cv xuống 2
    model.fit(X_train_scaled, y_train)

elif model_choice == "Artificial Neural Network":
    model = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

elif model_choice == "Stacking":
    base_models = [
        ('linear', LinearRegression()),
        ('lasso', Lasso(alpha=1.0)),
        ('mlp', MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42))
    ]
    meta_model = LinearRegression()
    model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=None)  # cv=none để không chia nhỏ dữ liệu
    model.fit(X_train_scaled, y_train)

# Dự đoán và tính toán sai số
predictions = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
st.write(f"Mean Squared Error (MSE) của mô hình: {mse:.2f}")

# Form nhập dữ liệu để dự đoán
st.subheader("Nhập thông tin để dự đoán chất lượng rượu")
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, value=0.7)
citric_acid = st.number_input("Citric Acid", min_value=0.0, value=0.0)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, value=1.9)
chlorides = st.number_input("Chlorides", min_value=0.0, value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, value=34.0)
density = st.number_input("Density", min_value=0.0, value=0.9978)
pH = st.number_input("pH", min_value=0.0, value=3.51)
sulphates = st.number_input("Sulphates", min_value=0.0, value=0.56)
alcohol = st.number_input("Alcohol", min_value=0.0, value=9.4)

# Dự đoán khi người dùng nhập thông tin
if st.button("Dự đoán chất lượng"):
    input_data = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                                free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]],
                              columns=X.columns)
    input_data_scaled = scaler.transform(input_data)
    predicted_quality = model.predict(input_data_scaled)[0]
    st.write(f"Chất lượng dự đoán: {predicted_quality:.2f}")
