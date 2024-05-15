import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def load_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        ad_dataset = pd.read_csv(file_path)
        return ad_dataset
    else:
        return None

def train_model(ad_dataset):
    # Strip whitespaces from column names
    ad_dataset.columns = ad_dataset.columns.str.strip()
    
    numerical_features = ['DailyTimeSpentonSite', 'Age', 'AreaIncome', 'DailyInternetUsage']
    X = ad_dataset[numerical_features]
    y = ad_dataset['Clicked on Ad']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    return model, scaler

def predict_ad_click(model, scaler, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    return prediction[0]

def on_predict_click():
    input_data = [float(entry.get()) for entry in entry_fields]
    prediction = predict_ad_click(model, scaler, input_data)
    result_label.config(text=f"Prediction: {'Clicked on Ad' if prediction == 1 else 'Not Clicked on Ad'}")

def on_load_data_click():
    global model, scaler
    ad_dataset = load_data()
    if ad_dataset is not None:
        model, scaler = train_model(ad_dataset)
        messagebox.showinfo("Info", "Data loaded and model trained successfully.")

def main():
    global entry_fields, result_label, model, scaler
    root = tk.Tk()
    root.title("Ad Data Analysis")

    entry_fields = []
    for i, feature in enumerate(['DailyTimeSpentonSite', 'Age', 'AreaIncome', 'DailyInternetUsage']):
        label = tk.Label(root, text=feature)
        label.grid(row=i, column=0, padx=10, pady=5)
        entry = tk.Entry(root)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entry_fields.append(entry)

    predict_button = tk.Button(root, text="Predict", command=on_predict_click)
    predict_button.grid(row=len(entry_fields), columnspan=2, pady=10)

    load_button = tk.Button(root, text="Load Data", command=on_load_data_click)
    load_button.grid(row=len(entry_fields)+1, columnspan=2, pady=10)

    result_label = tk.Label(root, text="")
    result_label.grid(row=len(entry_fields)+2, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
