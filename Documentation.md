## Untuk Load
```
import joblib

anfis = joblib.load('anfis_model.pkl')

scaler = joblib.load('scaler.pkl')

print("Model berhasil di-load")
```

## Prediksi Data
```
# contoh input baru
data_baru = [[120, 35, 28, 0]]

# scaling
data_baru_scaled = scaler.transform(data_baru)

# prediksi
prediksi = anfis.predict(data_baru_scaled)

print(prediksi)
```

## Probabilitas
```
prob = anfis.predict_proba(data_baru_scaled)

print(prob)
```


