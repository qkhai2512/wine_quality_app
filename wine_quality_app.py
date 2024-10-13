import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error



st.set_page_config(page_title="Dự Đoán Chất Lượng Rượu Vang", page_icon=":wine_glass:")

# Tải dữ liệu
def load_data():
    # Dữ liệu mẫu dựa trên hình ảnh bạn cung cấp
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
model_choice = st.selectbox("Chọn mô hình dự đoán", ["Linear Regression", "Lasso Regression"])

# Tách dữ liệu thành train/test
X = data.drop("quality", axis=1)
y = data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình dựa trên lựa chọn
if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Lasso Regression":
    model = Lasso(alpha=0.1)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Tính toán sai số
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
    predicted_quality = model.predict(input_data)[0]
    st.write(f"Chất lượng dự đoán: {predicted_quality:.2f}")

