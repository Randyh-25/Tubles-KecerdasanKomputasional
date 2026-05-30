# ============================================================
# app.py — ANFIS Diabetes Risk Detection System
# Kompatibel dengan: anfis_bundle.pkl (ANFISHybrid)
# Deploy: streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import itertools

# ─────────────────────────────────────────────────────────────
# BOILERPLATE CLASS ANFISHybrid
# Wajib ada agar joblib.load() bisa rebuild objek dari .pkl
# Hanya forward/predict — tanpa kode training
# ─────────────────────────────────────────────────────────────
FITUR = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']

class ANFIS:
    def __init__(self, n_mf=3, epochs=200, lr_premise=0.005,
                 l2_lambda=1e-4, patience=30):
        self.n_mf        = n_mf
        self.epochs      = epochs
        self.lr_premise  = lr_premise
        self.l2_lambda   = l2_lambda
        self.patience    = patience
        self.loss_hist   = []
        self._is_fitted  = False

    def _gaussmf(self, x, c, sigma):
        return np.exp(-0.5 * ((x - c) / (sigma + 1e-8)) ** 2)

    def _rule_strength(self, mu):
        n = mu.shape[0]
        strengths = np.ones((n, self.n_rules))
        for r, combo in enumerate(self.rule_combos):
            mu_stack = np.stack(
                [mu[:, i, mf_idx] for i, mf_idx in enumerate(combo)],
                axis=1)
            strengths[:, r] = mu_stack.min(axis=1)
        return strengths

    def _normalize(self, strengths):
        return strengths / (strengths.sum(axis=1, keepdims=True) + 1e-8)

    def _sigmoid(self, x):
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def _forward(self, X):
        n_samples, n_inputs = X.shape
        mu = np.zeros((n_samples, n_inputs, self.n_mf))
        for i in range(n_inputs):
            for k in range(self.n_mf):
                mu[:, i, k] = self._gaussmf(
                    X[:, i], self.centers[i, k], self.sigmas[i, k])
        strengths  = self._rule_strength(mu)
        w_norm     = self._normalize(strengths)
        X_aug      = np.hstack([X, np.ones((n_samples, 1))])
        consequents = X_aug @ self.C.T
        raw         = (w_norm * consequents).sum(axis=1)
        output      = self._sigmoid(raw)
        return output, raw, w_norm, strengths, mu

    def predict_proba(self, X):
        assert self._is_fitted, 'Model belum di-fit!'
        output, _, _, _, _ = self._forward(X)
        return output

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_membership(self, x_single):
        labels = ['Low', 'Medium', 'High']
        result = {}
        for i, feat in enumerate(FITUR):
            result[feat] = {
                lbl: float(self._gaussmf(
                    x_single[i], self.centers[i, k], self.sigmas[i, k]))
                for k, lbl in enumerate(labels)
            }
        return result


# ─────────────────────────────────────────────────────────────
# LOAD BUNDLE
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    bundle = joblib.load('anfis_bundle.pkl')
    return (bundle['model'],
            bundle['scaler'],
            bundle['thresh_low'],
            bundle['thresh_high'])


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def get_risk_category(score, thresh_low, thresh_high):
    if score < thresh_low:
        return "LOW",    "#1b5e20", "#d4edda", "🟢", "Risiko Rendah"
    elif score < thresh_high:
        return "MEDIUM", "#e65100", "#fff3cd", "🟡", "Risiko Sedang"
    else:
        return "HIGH",   "#b71c1c", "#f8d7da", "🔴", "Risiko Tinggi"


LABEL_DESC = {
    'Glucose': {
        'Low':    'Normal (<100 mg/dL)',
        'Medium': 'Prediabetes (100–125 mg/dL)',
        'High':   'Hiperglikemia (≥126 mg/dL)',
    },
    'BMI': {
        'Low':    'Normal / Kurus (<25)',
        'Medium': 'Kelebihan Berat Badan (25–30)',
        'High':   'Obesitas (≥30)',
    },
    'Age': {
        'Low':    'Muda (<35 th)',
        'Medium': 'Paruh Baya (35–55 th)',
        'High':   'Lanjut Usia (>55 th)',
    },
    'DiabetesPedigreeFunction': {
        'Low':    'Riwayat Genetik Rendah (<0.3)',
        'Medium': 'Riwayat Genetik Sedang (0.3–0.7)',
        'High':   'Riwayat Genetik Tinggi (>0.7)',
    },
}

MF_COLOR = {'Low': '#4CAF50', 'Medium': '#FF9800', 'High': '#F44336'}


def plot_radar(memberships, risk_score, category, color_dark,
               thresh_low, thresh_high):
    WEIGHTS = {'Low': 0.1, 'Medium': 0.5, 'High': 0.9}
    risk_per_feat = [
        sum(memberships[f][lbl] * WEIGHTS[lbl]
            for lbl in ['Low', 'Medium', 'High'])
        for f in FITUR
    ]
    short  = ['Glucose', 'BMI', 'Age', 'DPF']
    N      = len(FITUR)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values  = risk_per_feat + risk_per_feat[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, values, linewidth=2.5, color=color_dark)
    ax.fill(angles, values, alpha=0.25, color=color_dark)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short, fontsize=12, fontweight='bold', color='white')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.0'],
                       fontsize=7, color='grey')
    ax.grid(True, alpha=0.3, color='grey')
    ax.spines['polar'].set_color('grey')
    ax.set_title(
        f"Profil Risiko Pasien\n"
        f"Skor: {risk_score:.4f}  |  {category}\n"
        f"Threshold: {thresh_low:.3f} / {thresh_high:.3f}",
        fontweight='bold', fontsize=10, pad=18, color=color_dark
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLE
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANFIS Diabetes Risk Detection",
    page_icon="🩺",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.header-box {
    background: linear-gradient(135deg, #1565C0, #0288D1);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    color: white;
    margin-bottom: 1.5rem;
}
.header-box h1 { font-size: 1.75rem; margin: 0; font-weight: 700; }
.header-box p  { margin: 0.3rem 0 0; opacity: 0.88; font-size: 0.9rem; }

.result-box {
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    text-align: center;
    margin: 0.8rem 0;
}
.result-box .cat  { font-size: 1.9rem; font-weight: 700; }
.result-box .skor { font-size: 0.92rem; margin-top: 0.4rem; opacity: 0.85; }

.feat-row {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.6rem 0.9rem;
    border-radius: 8px;
    background: #1e2130;
    margin-bottom: 0.4rem;
    font-size: 0.85rem;
}
.mf-badge {
    border-radius: 20px;
    padding: 0.18rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    white-space: nowrap;
    margin-top: 2px;
}
.info-note {
    background: #1a2744;
    border-left: 4px solid #1565C0;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.85rem;
    color: #90caf9;
    margin-top: 1rem;
    line-height: 1.7;
}
.threshold-note {
    background: #1a1a2e;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.8rem;
    color: #78909c;
    margin-bottom: 1rem;
    text-align: center;
}
.method-badge {
    display: inline-block;
    background: #1b3a1b;
    border: 1px solid #2e7d32;
    border-radius: 20px;
    padding: 0.2rem 0.9rem;
    font-size: 0.78rem;
    color: #81c784;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🩺 ANFIS Diabetes Risk Detection</h1>
    <p>Adaptive Neuro-Fuzzy Inference System — Hybrid Learning (LSE + Gradient Descent)</p>
</div>
""", unsafe_allow_html=True)

# Load bundle
try:
    model, scaler, thresh_low, thresh_high = load_bundle()
except FileNotFoundError:
    st.error(
        "❌ File `anfis_bundle.pkl` tidak ditemukan. "
        "Pastikan file ada satu folder dengan `app.py`."
    )
    st.stop()

# Badge metode
st.markdown("""
<div style="text-align:center;">
    <span class="method-badge">
        ✅ Hybrid Learning &nbsp;|&nbsp;
        LSE untuk Consequent &nbsp;|&nbsp;
        Gradient Descent untuk Premise &nbsp;|&nbsp;
        Minimum T-norm &nbsp;|&nbsp;
        81 Rules (3⁴)
    </span>
</div>
""", unsafe_allow_html=True)

# Info threshold adaptif
st.markdown(
    f"""
    <div class="threshold-note">
        Threshold adaptif P33/P66 dari distribusi output model &nbsp;|&nbsp;
        🟢 LOW &lt; {thresh_low:.4f} &nbsp;|&nbsp;
        🟡 MEDIUM {thresh_low:.4f} – {thresh_high:.4f} &nbsp;|&nbsp;
        🔴 HIGH &gt; {thresh_high:.4f}
    </div>
    """,
    unsafe_allow_html=True
)

# ── Input pasien ─────────────────────────────────────────────
st.markdown("### 📋 Data Pasien")
st.caption("Masukkan nilai 4 fitur klinis pasien.")

col1, col2 = st.columns(2)
with col1:
    glucose = st.number_input(
        "🩸 Glucose (mg/dL)",
        min_value=50.0, max_value=250.0, value=117.0, step=1.0,
        help="Normal <100 | Prediabetes 100–125 | Diabetes ≥126"
    )
    age = st.number_input(
        "🎂 Age (tahun)",
        min_value=18, max_value=100, value=33, step=1,
        help="Usia pasien dalam tahun"
    )
with col2:
    bmi = st.number_input(
        "⚖️ BMI (kg/m²)",
        min_value=10.0, max_value=70.0, value=26.5, step=0.1,
        help="Normal <25 | Overweight 25–30 | Obese ≥30"
    )
    dpf = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0, value=0.35, step=0.01,
        help="Rendah <0.3 | Sedang 0.3–0.7 | Tinggi >0.7"
    )

st.markdown("---")
clicked = st.button(
    "Prediksi Risiko Diabetes",
    type="primary",
    use_container_width=True
)

if clicked:
    # Inferensi
    x_raw    = np.array([[glucose, bmi, age, dpf]])
    x_scaled = scaler.transform(x_raw)
    score    = float(model.predict_proba(x_scaled)[0])
    memb     = model.get_membership(x_scaled[0])
    cat, color_dark, color_bg, emoji, label_id = get_risk_category(
        score, thresh_low, thresh_high)

    # ── Kotak hasil utama ────────────────────────────────────
    st.markdown(
        f"""
        <div class="result-box"
             style="background:{color_bg}; border: 2px solid {color_dark};">
            <div class="cat" style="color:{color_dark};">
                {emoji} {cat} — {label_id}
            </div>
            <div class="skor" style="color:{color_dark};">
                Skor ANFIS Hybrid: <strong>{score:.4f}</strong>
                &nbsp;|&nbsp;
                Threshold: {thresh_low:.4f} / {thresh_high:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Risk meter ───────────────────────────────────────────
    # Normalisasi bar ke range aktual output model
    bar_min = thresh_low * 0.6
    bar_max = thresh_high * 1.4
    pct     = min(max((score - bar_min) / (bar_max - bar_min) * 100, 3), 97)
    st.markdown("**Risk Score Meter**")
    st.markdown(
        f"""
        <div style="background:#2d2d2d; border-radius:50px;
                    height:26px; overflow:hidden; margin-bottom:1rem;">
            <div style="width:{pct:.1f}%; background:{color_dark};
                        height:100%; border-radius:50px;
                        display:flex; align-items:center;
                        justify-content:flex-end; padding-right:12px;
                        color:white; font-weight:700; font-size:0.82rem;
                        transition: width 0.5s;">
                {score:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Dua kolom: Radar + Membership ────────────────────────
    col_r, col_m = st.columns([1, 1.05])

    with col_r:
        st.markdown("**Radar Chart Profil Fitur**")
        fig = plot_radar(memb, score, cat, color_dark,
                         thresh_low, thresh_high)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_m:
        st.markdown("**🔬 Fuzzy Membership per Fitur**")
        for feat in FITUR:
            dom    = max(memb[feat], key=memb[feat].get)
            desc   = LABEL_DESC[feat][dom]
            bcolor = MF_COLOR[dom]

            bars = "".join([
                f"<div style='display:inline-block;"
                f"width:{memb[feat][lbl]*55:.0f}px; height:7px;"
                f"background:{MF_COLOR[lbl]}; border-radius:3px;"
                f"margin-right:2px;'></div>"
                for lbl in ['Low', 'Medium', 'High']
            ])

            st.markdown(
                f"""
                <div class="feat-row">
                    <div style="min-width:130px; font-weight:600;
                                color:#cfd8dc; padding-top:2px;">
                        {feat}
                    </div>
                    <div>
                        <span class="mf-badge"
                              style="background:{bcolor};">{dom}</span>
                        <div style="color:#90a4ae; font-size:0.8rem;
                                    margin-top:4px;">{desc}</div>
                        <div style="margin-top:5px;">{bars}</div>
                        <div style="font-size:0.73rem; color:#607d8b;
                                    margin-top:3px;">
                            L:{memb[feat]['Low']:.3f} &nbsp;
                            M:{memb[feat]['Medium']:.3f} &nbsp;
                            H:{memb[feat]['High']:.3f}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ── Narasi kesimpulan ────────────────────────────────────
    narratives = [
        f"<b>{f}</b>: {LABEL_DESC[f][max(memb[f], key=memb[f].get)]}"
        for f in FITUR
    ]
    st.markdown(
        f"""
        <div class="info-note">
            <b>Kesimpulan Fuzzy Reasoning:</b><br>
            {' &nbsp;|&nbsp; '.join(narratives)}<br><br>
            Skor akhir ANFIS Hybrid <b>{score:.4f}</b> dikategorikan sebagai
            <b style="color:{color_dark};">{label_id} ({cat})</b>
            berdasarkan threshold adaptif P33/P66 dari distribusi
            output model pada seluruh data training dan testing.
        </div>
        """,
        unsafe_allow_html=True
    )

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "ANFISHybrid | LSE + Gradient Descent | "
    "Minimum T-norm | Gaussian MF | 4 Fitur | 81 Rules | "
    "Pima Indians Diabetes Dataset (UCI)"
)