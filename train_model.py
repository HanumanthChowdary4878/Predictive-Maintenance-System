#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from sklearn.model_selection import train_test_split, cross_val_score


# In[4]:


from sklearn.ensemble import RandomForestRegressor


# In[5]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[6]:


import joblib


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


import seaborn as sns


# In[9]:


df = pd.read_csv("data.csv")


# In[10]:


df.rename(columns={
    "Wind Speed (m/s)": "Wind Speed",
    "Theoretical_Power_Curve (KWh)": "Theoretical Power",
    "Wind Direction (°)": "Wind Direction",
    "LV ActivePower (kW)": "Active Power"
}, inplace=True)


# In[11]:


if 'Date/Time' in df.columns:
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M', errors='coerce')


# In[12]:


numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()


# In[13]:


df_numeric = df[numeric_columns]


# In[14]:


df.shape


# In[15]:


df_numeric.describe()


# In[16]:


missing_values = df_numeric.isnull().sum()


# In[17]:


missing_percentage = (missing_values / len(df_numeric)) * 100


# In[18]:


missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Percentage': missing_percentage
})


# In[19]:


print(missing_df)


# In[20]:


correlation_matrix = df_numeric.corr()


# In[21]:


target_correlations = correlation_matrix["Active Power"].sort_values(ascending=False)


# In[22]:


target_correlations


# In[23]:


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.3f', square=True, linewidths=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()


# In[24]:


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(df_numeric["Wind Speed"], df_numeric["Active Power"], alpha=0.5, s=1)
axes[0, 0].set_xlabel('Wind Speed')
axes[0, 0].set_ylabel('Active Power')
axes[0, 0].set_title('Wind Speed vs Active Power')

axes[0, 1].scatter(df_numeric["Theoretical Power"], df_numeric["Active Power"], alpha=0.5, s=1)
axes[0, 1].set_xlabel('Theoretical Power')
axes[0, 1].set_ylabel('Active Power')
axes[0, 1].set_title('Theoretical Power vs Active Power')

axes[1, 0].scatter(df_numeric["Wind Direction"], df_numeric["Active Power"], alpha=0.5, s=1)
axes[1, 0].set_xlabel('Wind Direction')
axes[1, 0].set_ylabel('Active Power')
axes[1, 0].set_title('Wind Direction vs Active Power')

axes[1, 1].hist(df_numeric["Active Power"], bins=50, edgecolor='black')
axes[1, 1].set_xlabel('Active Power')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Active Power Distribution')

plt.tight_layout()
plt.show()


# In[25]:


X = df[["Wind Speed", "Theoretical Power", "Wind Direction"]]


# In[26]:


y = df["Active Power"]


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[28]:


print(f"Training set size: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")


# In[29]:


print(f"Test set size: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")


# In[30]:


model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)


# In[31]:


model.fit(X_train, y_train)


# In[32]:


cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                            scoring='r2', n_jobs=-1)


# In[33]:


print(f"Mean CV R² score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")


# In[34]:


y_train_pred = model.predict(X_train)


# In[35]:


y_test_pred = model.predict(X_test)


# In[36]:


train_mse = mean_squared_error(y_train, y_train_pred)


# In[37]:


train_rmse = np.sqrt(train_mse)


# In[38]:


train_mae = mean_absolute_error(y_train, y_train_pred)


# In[39]:


train_r2 = r2_score(y_train, y_train_pred)


# In[40]:


test_mse = mean_squared_error(y_test, y_test_pred)


# In[41]:


test_rmse = np.sqrt(test_mse)


# In[42]:


test_mae = mean_absolute_error(y_test, y_test_pred)


# In[43]:


test_r2 = r2_score(y_test, y_test_pred)


# In[44]:


print(f"Training - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")


# In[45]:


print(f"Test - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")


# In[46]:


overfitting_gap = train_r2 - test_r2


# In[47]:


print(f"\nOverfitting gap: {overfitting_gap:.4f}")


# In[48]:


feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)


# In[49]:


feature_importance


# In[50]:


plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest Model')
plt.tight_layout()
plt.show()


# In[51]:


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=1)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
             'r--', lw=2)
axes[0].set_xlabel('Actual Active Power')
axes[0].set_ylabel('Predicted Active Power')
axes[0].set_title(f'Training Set Predictions (R² = {train_r2:.4f})')

axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=1)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2)
axes[1].set_xlabel('Actual Active Power')
axes[1].set_ylabel('Predicted Active Power')
axes[1].set_title(f'Test Set Predictions (R² = {test_r2:.4f})')

plt.tight_layout()
plt.show()


# In[52]:


train_residuals = y_train - y_train_pred


# In[53]:


test_residuals = y_test - y_test_pred


# In[54]:


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(y_train_pred, train_residuals, alpha=0.5, s=1)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Training Set Residuals')

axes[0, 1].scatter(y_test_pred, test_residuals, alpha=0.5, s=1)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted Values')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Test Set Residuals')

axes[1, 0].hist(train_residuals, bins=50, edgecolor='black')
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Training Set Residual Distribution')

axes[1, 1].hist(test_residuals, bins=50, edgecolor='black')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Test Set Residual Distribution')

plt.tight_layout()
plt.show()


# In[55]:


print(f"Training residuals mean: {train_residuals.mean():.4f}")


# In[56]:


print(f"Training residuals std: {train_residuals.std():.4f}")


# In[57]:


print(f"Test residuals mean: {test_residuals.mean():.4f}")


# In[58]:


print(f"Test residuals std: {test_residuals.std():.4f}")


# In[59]:


joblib.dump(model, "model.pkl")


# In[60]:


print("Model saved as model.pkl")


# In[61]:


model_metadata = {
    'features': list(X.columns),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'train_r2': train_r2,
    'test_r2': test_r2,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'feature_importance': feature_importance.to_dict('records')
}


# In[62]:


joblib.dump(model_metadata, "model_metadata.pkl")


# In[63]:


print("Model metadata saved as model_metadata.pkl")


# In[ ]:




