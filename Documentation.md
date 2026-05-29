## Cara Save Model
```python
import joblib

# save model
joblib.dump(anfis, 'anfis_model.pkl') # Nama bebas, yg penting diingat

# save scaler
joblib.dump(scaler, 'scaler.pkl') # Nama bebas, yg penting diingat

print("Model berhasil disimpan")
```

## Untuk Load
```python
import joblib

anfis = joblib.load('anfis_model.pkl')

scaler = joblib.load('scaler.pkl')

print("Model berhasil di-load")
```

## Prediksi Data
```python
# contoh input baru
data_baru = [[120, 35, 28, 0]]

# scaling
data_baru_scaled = scaler.transform(data_baru)

# prediksi
prediksi = anfis.predict(data_baru_scaled)

print(prediksi)
```

## Probabilitas
```python
prob = anfis.predict_proba(data_baru_scaled)

print(prob)
```


