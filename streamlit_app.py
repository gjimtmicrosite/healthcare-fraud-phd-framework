
import streamlit as st
import pandas as pd
import numpy as np
import hashlib, json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Healthcare Insurance Fraud Prevention Framework",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top:1.2rem; padding-bottom:2rem;}
.main-title {border-bottom:2px solid rgba(120,120,120,.35); padding-bottom:.8rem; margin-bottom:1.2rem;}
.note-box {border-left:4px solid rgba(100,100,100,.65); padding:.7rem 1rem; background:rgba(120,120,120,.08); border-radius:4px;}
div[data-testid="stMetric"] {border:1px solid rgba(120,120,120,.25); padding:12px; border-radius:8px;}
[data-testid="stSidebar"] {border-right:1px solid rgba(120,120,120,.20);}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
<h1>Blockchain-Based Framework for Healthcare Insurance Fraud Prevention</h1>
<p>Doctoral Research Prototype: Provider-Level Fraud-Risk Analysis, Explainability and Evidence Integrity</p>
</div>
""", unsafe_allow_html=True)

PUBLISHED = {
    "Accuracy":0.940, "Precision":0.702, "Recall":0.630,
    "F1-score":0.664, "ROC-AUC":0.951, "Average Precision":0.733
}

DATASET_PROFILE = pd.DataFrame({
    "Dataset":["Provider / Training","Beneficiary","Inpatient","Outpatient"],
    "Records":[5410,138556,40474,517737],
    "Attributes":[2,25,30,27]
})

FEATURE_GROUPS = [
    "Length-of-stay statistics",
    "Inpatient claim reimbursement aggregates",
    "Outpatient claim reimbursement aggregates",
    "Deductible amount aggregates",
    "Beneficiary age and chronic-condition indicators",
    "Diagnosis-code frequency features",
    "Procedure-code frequency features",
    "Provider-level claim-count features",
    "Other engineered provider-level numerical features"
]

def demo_data(n=4057, seed=42):
    rng = np.random.default_rng(seed)
    y = np.array([1]*379 + [0]*(n-379))
    rng.shuffle(y)
    names = [
        "ip__LOS_max","ip__LOS_std","ip__LOS_count","ip__LOS_mean","ip__LOS_median",
        "op__LOS_count","ip__InscClaimAmtReimbursed_sum","ip__InscClaimAmtReimbursed_mean",
        "ip__InscClaimAmtReimbursed_std","op__InscClaimAmtReimbursed_sum",
        "op__InscClaimAmtReimbursed_mean","op__InscClaimAmtReimbursed_std",
        "ip__DeductibleAmtPaid_sum","ip__DeductibleAmtPaid_mean",
        "op__DeductibleAmtPaid_sum","op__DeductibleAmtPaid_mean","op__DeductibleAmtPaid_std",
        "bene__age","bene__ChronicCond_Depression","bene__ChronicCond_ObstrPulmonary",
        "bene__ChronicCond_stroke"
    ] + [f"engineered_feature_{i:02d}" for i in range(22,56)]
    X = rng.normal(size=(n,55))
    for j in [0,2,6,9,12,17,18,19,20]:
        X[:,j] += y*rng.uniform(.45,1.0)
    d = pd.DataFrame(X, columns=names)
    d["PotentialFraud"] = y
    d["Provider"] = [f"PRV{100000+i}" for i in range(n)]
    return d

@st.cache_resource
def fitted_demo():
    d = demo_data()
    features = [c for c in d.columns if c not in ["PotentialFraud","Provider"]]
    Xtr, Xte, ytr, yte = train_test_split(
        d[features], d["PotentialFraud"], test_size=.25,
        stratify=d["PotentialFraud"], random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=220, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(Xtr,ytr)
    return d,features,model,Xte,yte

def hash_obj(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(",",":")).encode()
    return hashlib.sha256(raw).hexdigest()

def merkle_root(hashes):
    level = hashes[:]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256((level[i]+level[i+1]).encode()).hexdigest()
            for i in range(0,len(level),2)
        ]
    return level[0]

with st.sidebar:
    st.markdown("### Research Navigation")
    page = st.radio("", [
        "Research Overview",
        "Dataset Profile",
        "Methodology",
        "Provider Risk Analysis",
        "Explainability",
        "Blockchain Verification",
        "Experimental Results",
        "Research Objectives"
    ])
    st.divider()
    st.caption("Research scope")
    st.write("The framework is a decision-support prototype. Fraud-risk prediction does not independently establish legal guilt, and blockchain verification confirms evidence integrity rather than factual truth.")

d, features, model, Xte, yte = fitted_demo()

if page == "Research Overview":
    st.subheader("Research Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Reported ROC-AUC","0.951")
    c2.metric("Reported Accuracy","94.0%")
    c3.metric("Reported AP","0.733")
    c4.metric("Reported Blockchain Latency","120–250 ms")
    st.markdown("### Proposed Research Framework")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR;
      node [shape=box style="rounded,filled" fillcolor="#f4f4f4" color="#777777"];
      A [label="Healthcare and Insurance Records"];
      B [label="Data Integration and Preprocessing"];
      C [label="Provider-Level Feature Representation"];
      D [label="Machine Learning Fraud-Risk Prediction"];
      E [label="Explainability"];
      F [label="Evidence Artefact"];
      G [label="SHA-256 / Merkle Verification"];
      H [label="Permissioned Blockchain"];
      A->B->C->D->E->F->G->H;
    }""")
    st.markdown("""
    <div class="note-box"><b>Research contribution:</b> The work integrates provider-level fraud-risk prediction,
    interpretable explanations and cryptographically verifiable evidence within a single
    healthcare-insurance fraud prevention framework.</div>
    """, unsafe_allow_html=True)

elif page == "Dataset Profile":
    st.subheader("Dataset Profile")
    st.write("The experimental study uses four linked healthcare-insurance data tables. After integration and feature engineering, the analytical representation contains 55 numerical provider-level features.")
    st.dataframe(DATASET_PROFILE, use_container_width=True, hide_index=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Final Numerical Features","55")
    c2.metric("Potentially Fraudulent Providers","379")
    c3.metric("Non-Fraudulent Providers","3,678")
    st.markdown("### Original Provider-Level Class Distribution")
    cls = pd.DataFrame({"Class":["Potentially Fraudulent","Non-Fraudulent"],"Providers":[379,3678]})
    fig,ax = plt.subplots(figsize=(7,4))
    ax.bar(cls["Class"], cls["Providers"])
    ax.set_ylabel("Number of Providers")
    ax.set_title("Original Provider-Level Class Distribution")
    st.pyplot(fig)
    st.markdown("### Feature Engineering Summary")
    for item in FEATURE_GROUPS:
        st.write("• " + item)
    st.markdown("### Class-Imbalance Treatment")
    st.write("SMOTE-Tomek is applied only to the training data. The reported balanced training distribution is 3,675 fraudulent and 3,675 non-fraudulent observations.")
    st.markdown("### Validation Strategy")
    st.write("Provider-aware GroupKFold cross-validation (k = 5) is used to reduce provider leakage between training and validation folds.")
    st.caption("For the live web interface, a lightweight demonstration dataset is used for interactive risk scoring. Published dataset statistics and experimental results are presented separately from the interface demonstration.")

elif page == "Methodology":
    st.subheader("Research Methodology")
    st.markdown("### Data Processing")
    st.write("Duplicate removal, missing-value treatment, integration of inpatient, outpatient, beneficiary and provider information, numerical normalization, feature engineering and training-only resampling.")
    st.markdown("### Principal Learning Models")
    st.dataframe(pd.DataFrame({
        "Model":["XGBoost","LightGBM","Stacking Ensemble"],
        "Role":[
            "Gradient-boosted fraud-risk classifier",
            "Gradient-boosted fraud-risk classifier",
            "Combines XGBoost and LightGBM outputs using Logistic Regression as meta-learner"
        ]
    }), use_container_width=True, hide_index=True)
    st.markdown("### Selected Hyperparameters")
    st.dataframe(pd.DataFrame({
        "Model":["XGBoost","LightGBM","Logistic Regression Meta-Learner"],
        "Configuration":[
            "300 estimators; learning rate 0.01; max depth 6; subsample 0.8; colsample_bytree 0.8; reg_alpha 0.1; reg_lambda 1.0; gamma 0.2",
            "250 estimators; learning rate 0.01; num_leaves 31; max depth 8; subsample 0.8; colsample_bytree 0.8; min_child_samples 20; reg_alpha 0.1; reg_lambda 1.0",
            "L2 penalty; C 1.0; lbfgs solver; max_iter 500"
        ]
    }), use_container_width=True, hide_index=True)
    st.markdown("### Explainability and Evidence Integrity")
    st.write("SHAP is used to interpret model predictions. Fraud-related analytical evidence is hashed using SHA-256 and organized through Merkle-tree verification before anchoring to a permissioned blockchain using pBFT-style validation.")

elif page == "Provider Risk Analysis":
    st.subheader("Provider-Level Fraud-Risk Analysis")
    idx = st.slider("Provider record", 0, min(500,len(Xte)-1), 25)
    row = Xte.iloc[idx:idx+1]
    risk = float(model.predict_proba(row)[0,1])
    provider = d.loc[row.index[0],"Provider"]
    c1,c2,c3 = st.columns(3)
    c1.metric("Provider ID", provider)
    c2.metric("Fraud-Risk Score", f"{risk:.3f}")
    c3.metric("Review Category", "Higher Risk" if risk >= .5 else "Lower Risk")
    st.progress(float(min(max(risk,0),1)))
    st.dataframe(pd.DataFrame({
        "Feature": row.columns[:12],
        "Observed Value": row.iloc[0,:12].round(3).values
    }), use_container_width=True, hide_index=True)
    st.info("Interpretation: the score is a fraud-risk decision-support output and should be reviewed together with explanation and supporting claim evidence.")

elif page == "Explainability":
    st.subheader("Provider-Level Explanation")
    idx = st.slider("Provider record", 0, min(500,len(Xte)-1), 25, key="explain_idx")
    row = Xte.iloc[idx:idx+1]
    base = float(model.predict_proba(row)[0,1])
    impacts = {}
    for col in features:
        altered = row.copy()
        altered[col] = Xte[col].median()
        impacts[col] = base - float(model.predict_proba(altered)[0,1])
    imp = pd.Series(impacts).sort_values(key=np.abs, ascending=False).head(12).sort_values()
    fig,ax = plt.subplots(figsize=(8,5))
    ax.barh(imp.index, imp.values)
    ax.set_xlabel("Change in predicted risk after neutralizing feature")
    ax.set_title("Local Feature Influence")
    st.pyplot(fig)
    st.markdown("""
    <div class="note-box">In the published study, SHAP is the principal explainability method.
    The web interface uses a lightweight local influence view for stable deployment, while the reported SHAP
    stability results are shown under Experimental Results.</div>
    """, unsafe_allow_html=True)

elif page == "Blockchain Verification":
    st.subheader("Blockchain Evidence Verification")
    idx = st.slider("Provider record", 0, min(500,len(Xte)-1), 25, key="block_idx")
    row = Xte.iloc[idx:idx+1]
    risk = float(model.predict_proba(row)[0,1])
    provider = d.loc[row.index[0],"Provider"]
    evidence = {
        "provider_id": provider,
        "fraud_risk_score": round(risk,6),
        "evidence_type": "provider-level fraud-risk decision support",
        "patient_data_on_chain": False
    }
    evidence_hash = hash_obj(evidence)
    root = merkle_root([evidence_hash] + [hashlib.sha256(f"evidence-{i}".encode()).hexdigest() for i in range(1,8)])
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Off-Chain Evidence Record")
        st.json(evidence)
    with c2:
        st.markdown("#### Cryptographic Commitment")
        st.code(evidence_hash, language=None)
        st.markdown("**Merkle Root**")
        st.code(root, language=None)
    st.success("Verification status: evidence record matches the stored cryptographic commitment.")
    if st.checkbox("Modify the evidence record for verification test"):
        changed = dict(evidence)
        changed["fraud_risk_score"] = round(min(risk + .10, 1),6)
        st.error("Verification status: mismatch detected. The modified evidence generates a different hash.")
        st.code(hash_obj(changed), language=None)

elif page == "Experimental Results":
    st.subheader("Published Experimental Results")
    st.caption("Reported values from the published experimental study.")
    cols = st.columns(6)
    for col,(k,v) in zip(cols,PUBLISHED.items()):
        col.metric(k, f"{v:.3f}")
    fig,ax = plt.subplots(figsize=(9,4))
    ax.bar(list(PUBLISHED.keys()), list(PUBLISHED.values()))
    ax.set_ylim(0,1)
    ax.set_ylabel("Reported Score")
    ax.set_title("Stacking Ensemble: Reported Performance")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)
    st.markdown("### Provider-Aware 5-Fold Cross-Validation")
    st.dataframe(pd.DataFrame({
        "Model":["XGBoost","LightGBM","Stacking"],
        "ROC-AUC":["0.937 ± 0.006","0.936 ± 0.007","0.951 ± 0.005"],
        "F1-score":["0.688 ± 0.012","0.680 ± 0.014","0.664 ± 0.010"]
    }), use_container_width=True, hide_index=True)
    st.markdown("### Explainability Stability")
    st.dataframe(pd.DataFrame({
        "Measure":["Mean Spearman rank correlation","Top-10 feature overlap","Correlation of aggregated SHAP contribution with predicted fraud probability"],
        "Reported Result":["≈ 0.82","> 85%","≈ 0.42–0.55"]
    }), use_container_width=True, hide_index=True)
    st.markdown("### Blockchain Performance")
    st.dataframe(pd.DataFrame({
        "Measure":["Transaction latency","Throughput","Block confirmation","SHA-256 hash generation","Transaction creation","Transaction submission","pBFT validation","Merkle proof generation"],
        "Reported Result":["120–250 ms","800–1200 TPS","< 1 second","5–10 ms","20–40 ms","50–100 ms","80–150 ms","2–5 ms"]
    }), use_container_width=True, hide_index=True)
    st.info("The voting classifier achieved the highest reported Average Precision (0.746), while the stacking ensemble provided a strong overall balance with ROC-AUC 0.951.")

elif page == "Research Objectives":
    st.subheader("Research Objectives and Evidence")
    st.dataframe(pd.DataFrame({
        "Research Objective":[
            "Objective 1: Analyze existing challenges and vulnerabilities in managing health and insurance records",
            "Objective 2: Design a blockchain-based framework to enhance financial security and prevent insurance frauds",
            "Objective 3: Performance evaluation and comparison with existing ABHA framework"
        ],
        "Evidence in Research":[
            "Analysis of record-management vulnerabilities, fraud risks, integrity, transparency, privacy and accountability challenges",
            "Integrated provider-level ML, explainability and permissioned-blockchain evidence-integrity framework",
            "Experimental model/blockchain evaluation and functional/architectural comparison with the existing ABHA framework"
        ]
    }), use_container_width=True, hide_index=True)
    st.markdown("### Interpretation of Objective 3")
    st.write("The ABHA comparison is functional and architectural rather than a direct head-to-head operational performance benchmark. The proposed framework is positioned as complementary to the broader digital-health ecosystem.")

st.divider()
st.caption("Doctoral research prototype for academic demonstration. Interactive interface results are separated from published experimental benchmarks.")
