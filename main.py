import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import *

## Lấy dữ liệu
data = pd.read_csv('cleaned.csv')
# data = data.sample(1000) 
# data.info()

# Mô tả dữ liệu
data.replace({'gender':{0:'male',1:'female'}},inplace=True)
features = ['age','height','weight','duration','body_temp','heart_rate'] 
for i, col in enumerate(features): 
    plt.subplot(2, 3, i + 1) 
    sb.scatterplot(data=data, x=data[col],y=data.calories,hue=data.gender)
plt.show()
data.replace({'gender':{'male':0,'female':1}},inplace=True)

# Xác định dữ liệu đầu vào
x=data.drop(['user_id','calories'],axis=1).values
# Xác định dữ liệu đầu ra
y=data['calories'].values
# Chia dữ liệu cần học và dữ liệu cần kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
# In ra kích thước dữ liệu đầu vào và tỷ lệ cần học và kiểm tra
print('Input:',x.shape,'Train:',x_train.shape,'Test:',x_test.shape)
## Đánh giá thuật toán
print("\n---Algorithm evaluation---")
# Khởi tạo đa thức bậc 3
poly_reg = PolynomialFeatures(degree=3)
# Đa thức bậc 3 dữ liệu đầu vào cần học
X_poly = poly_reg.fit_transform(x_train)
# In ra kích thước đầu vào sau khi đa thức bậc 3
print('Degree 3:',x_train.shape,'To',X_poly.shape)
# Khởi tạo hồi quy tuyến tính
pol_reg = LinearRegression()
# Xây dựng hồi quy
pol_reg.fit(X_poly, y_train)
# Kiểm tra sai số tuyệt đối trung bình đầu vào cần học
train_preds=pol_reg.predict(X_poly)
print('Training Error : ', mean_absolute_error(y_train, train_preds)) 
# Kiểm tra sai số tuyệt đối trung bình đầu vào cần kiểm tra đã đa thức bậc 3
val_preds=pol_reg.predict(poly_reg.fit_transform(x_test))
print('Validation Error : ', mean_absolute_error(y_test,val_preds)) 
# In ra hệ số xác định hồi quy
print('R2_score = ', r2_score(y_test, val_preds))
# In ra lỗi dư tối đa
print('Max_error = ',max_error(y_test,val_preds))
## Dự đoán
print("\n---Forecast---")
# Nhập dữ liệu cần dự đoán
while(1):
    features = ["gender {'male':0, 'female':1}",'age','height','weight','duration','body_temp','heart_rate']
    i=0
    while(i<len(features)):
        value = float(input(features[i]+": "))
        if(value > 0 and i != 0 or i == 0 and value in [0,1]):
            features[i] = value
            i+=1
    # Tính toán và in ra kết quả dự đoán
    y_pred = pol_reg.predict(poly_reg.fit_transform([features]))
    print('Calories burned: '+str(y_pred[0]))
    if(input('Next prediction? (y/n): ')=='n'): break

# so sánh các mô hình khác
models = [LinearRegression(),XGBRegressor(), RandomForestRegressor(), PolynomialFeatures(degree=3)] 
for i in range(len(models)):
    print(f'{models[i]} : ') 
    if(i==len(models)-1):
        models[0].fit(models[i].fit_transform(x_train), y_train)
        train_preds = models[0].predict(models[i].fit_transform(x_train))
        val_preds = models[0].predict(models[i].fit_transform(x_test))
    else:
        models[i].fit(x_train, y_train)
        train_preds = models[i].predict(x_train) 
        val_preds = models[i].predict(x_test) 
    print('Training Error : ', mean_absolute_error(y_train, train_preds)) 
    print('Validation Error : ', mean_absolute_error(y_test,val_preds)) 
    print('r2_score = ', r2_score(y_test,val_preds))
    print('max_error = ',max_error(y_test,val_preds))
    print() 
    
    plt.subplot(2, 2, i + 1) 
    prediction= pd.DataFrame({'actual Calories Burnt': y_test, 'predicted Calories Burnt': val_preds})
    sb.scatterplot(data=prediction, x='actual Calories Burnt',y='predicted Calories Burnt')
    plt.plot(y_test, y_test, color="red", linewidth=3)
plt.show()

# features = ['age','height','weight','duration','body_temp','heart_rate'] 
# for i, col in enumerate(features): 
#     print(col)
#     x=data[col].values.reshape(-1, 1)
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
#     models = [LinearRegression(),XGBRegressor(), RandomForestRegressor(), PolynomialFeatures(degree=3)] 
#     for i in range(len(models)): 
#         print(f'{models[i]} : ') 
#         if(i==len(models)-1):
#             models[0].fit(models[i].fit_transform(x_train), y_train)
#             train_preds = models[0].predict(models[i].fit_transform(x_train))
#             val_preds = models[0].predict(models[i].fit_transform(x_test))
#         else:
#             models[i].fit(x_train, y_train)
#             print(f'{models[i]} : ') 
#             train_preds = models[i].predict(x_train) 
#             val_preds = models[i].predict(x_test) 
#         print('Training Error : ', mean_absolute_error(y_train, train_preds)) 
#         print('Validation Error : ', mean_absolute_error(y_test,val_preds)) 
#         print('r2_score = ', r2_score(y_test,val_preds))
#         print('max_error = ',max_error(y_test,val_preds))
#         print() 

#         plt.subplot(2, 2, i + 1) 
#         plt.scatter(x_test, y_test, color="black")
#         plt.plot(x_test, val_preds, color="red", linewidth=3)
#     plt.show()  