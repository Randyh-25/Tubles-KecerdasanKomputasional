import joblib
import numpy as np
import streamlit as st
from itertools import product

class SimpleANFIS:
    def __init__(self, n_inputs, n_mfs=2, lr=0.01):
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.lr = lr
        self.centers = np.array([
            np.linspace(0.25, 0.75, n_mfs)
            for _ in range(n_inputs)
        ])
        self.sigmas = np.full((n_inputs, n_mfs), 0.2)

        self.rules = list(product(range(n_mfs), repeat=n_inputs))
        self.n_rules = len(self.rules)
        self.C = np.random.randn(self.n_rules) * 0.01
        self.loss_history = []

    def gaussian(self, x, c, s):
        return np.exp(-0.5 * ((x - c) / (s + 1e-8)) ** 2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        n_samples = X.shape[0]

        memberships = []
        for i in range(self.n_inputs):
            mf_values = []
            for j in range(self.n_mfs):
                mf = self.gaussian(
                    X[:, i],
                    self.centers[i, j],
                    self.sigmas[i, j],
                )
                mf_values.append(mf)
            memberships.append(np.array(mf_values).T)

        firing_strengths = []
        for rule in self.rules:
            fs = np.ones(n_samples)
            for feature_idx, mf_idx in enumerate(rule):
                fs *= memberships[feature_idx][:, mf_idx]
            firing_strengths.append(fs)

        firing_strengths = np.array(firing_strengths).T

        firing_sum = np.sum(firing_strengths, axis=1, keepdims=True)
        firing_sum = np.where(firing_sum == 0, 1e-8, firing_sum)
        normalized_fs = firing_strengths / firing_sum

        output_linear = np.dot(normalized_fs, self.C)
        output = self.sigmoid(output_linear)

        return output, normalized_fs

    def predict(self, X):
        output, _ = self.forward(X)
        return (output >= 0.5).astype(int)

    def predict_proba(self, X):
        output, _ = self.forward(X)
        return output


@st.cache_resource
def load_artifacts():
    model = joblib.load("anfis_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


def main():
    st.set_page_config(page_title="Prediksi Diabetes ANFIS", page_icon="🩺", layout="centered")
    st.title("Prediksi Diabetes dengan ANFIS")
    st.caption("Model memuat artefak dari file anfis_model.pkl dan scaler.pkl")

    try:
        model, scaler = load_artifacts()
    except Exception as err:
        st.error(f"Gagal memuat model/scaler: {err}")
        st.stop()

    st.subheader("Input Data Pasien")
    glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0, step=0.1)
    age = st.number_input("Age", min_value=1.0, max_value=120.0, value=28.0, step=1.0)
    dpf = st.number_input(
        "DiabetesPedigreeFunction",
        min_value=0.0,
        max_value=3.0,
        value=0.35,
        step=0.01,
        format="%.2f",
    )

    if st.button("Prediksi", type="primary"):
        x = np.array([[glucose, bmi, age, dpf]], dtype=float)
        x_scaled = scaler.transform(x)

        prob = model.predict_proba(x_scaled)
        prob = float(np.asarray(prob).reshape(-1)[0])

        pred = int(model.predict(x_scaled)[0])
        label = "Diabetes" if pred == 1 else "Tidak Diabetes"

        st.success(f"Hasil Prediksi: {label}")
        st.metric("Probabilitas Diabetes", f"{prob * 100:.2f}%")


if __name__ == "__main__":
    main()
