from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Memuat model yang telah dilatih dari file H5
model = load_model('best_model.h5')

# Inisialisasi standard scaler untuk data input
sc = StandardScaler()
# Asumsikan Anda memiliki akses ke data asli untuk menyesuaikan scaler
original_data = pd.read_csv('Churn_Modelling.csv', index_col='RowNumber')
original_data.drop(['CustomerId', 'Surname'], axis=1, inplace=True)
Geography_dummies = pd.get_dummies(prefix='Geo', data=original_data, columns=['Geography'])
Gender_dummies = Geography_dummies.replace(to_replace={'Gender': {'Female': 1, 'Male': 0}})
churn_data_encoded = Gender_dummies
X = churn_data_encoded.drop(['Exited'], axis=1)
sc.fit(X)

# Mendefinisikan rute home
@app.route('/')
def home():
    return render_template('index.html')

# Mendefinisikan rute prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan data dari permintaan POST
    data = request.get_json(force=True)
    
    # Mengonversi data menjadi array numpy dan menstandarkan
    input_data = np.array([data['features']])
    input_data_scaled = sc.transform(input_data)
    
    # Membuat prediksi
    prediction = model.predict(input_data_scaled)
    
    # Mengonversi prediksi menjadi daftar dan mengembalikannya sebagai JSON
    output = prediction[0].tolist()
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True)
