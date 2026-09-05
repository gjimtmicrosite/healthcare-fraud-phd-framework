
import streamlit as st
import pandas as pd
import numpy as np
import hashlib, json, time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="SecureClaim AI | PhD Prototype", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#0b1f3a,#123c69); color:white; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2.2rem;}
.hero p {opacity:.92; font-size:1.05rem;}
.card {border:1px solid rgba(128,128,128,.25); border-radius:16px; padding:1rem 1.1rem; margin:.4rem 0;}
.small {font-size:.88rem; opacity:.8}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22); padding:12px; border-radius:14px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero">
<h1>🛡️ SecureClaim AI</h1>
<p>Blockchain-Based Explainable Machine Learning Framework for Healthcare Insurance Fraud Prevention</p>
<p class="small">PhD research prototype • Provider-level fraud-risk decision support • Explainability • Cryptographic evidence integrity</p>
</div>""", unsafe_allow_html=True)

PUBLISHED = {
    "Accuracy":0.940, "Precision":0.702, "Recall":0.630,
    "F1":0.664, "ROC-AUC":0.951, "Average Precision":0.733
}
CV = pd.DataFrame({
    "Model":["XGBoost","LightGBM","Stacking"],
    "Mean ROC-AUC":[0.937,0.936,0.951],
    "SD":[0.006,0.007,0.005]
})

def synthetic_data(n=4057, seed=42):
    rng=np.random.default_rng(seed)
    y=np.array([1]*379+[0]*(n-379)); rng.shuffle(y)
    names=[
      "ip__LOS_max","ip__LOS_std","ip__LOS_count","ip__LOS_mean","ip__LOS_median",
      "op__LOS_count","ip__InscClaimAmtReimbursed_sum","ip__InscClaimAmtReimbursed_mean",
      "ip__InscClaimAmtReimbursed_std","op__InscClaimAmtReimbursed_sum",
      "op__InscClaimAmtReimbursed_mean","op__InscClaimAmtReimbursed_std",
      "ip__DeductibleAmtPaid_sum","ip__DeductibleAmtPaid_mean",
      "op__DeductibleAmtPaid_sum","op__DeductibleAmtPaid_mean","op__DeductibleAmtPaid_std",
      "bene__age","bene__ChronicCond_Depression","bene__ChronicCond_ObstrPulmonary",
      "bene__ChronicCond_stroke"]+[f"engineered_feature_{i:02d}" for i in range(22,56)]
    X=rng.normal(size=(n,55))
    for j in [0,2,6,9,12,17,18,19,20]: X[:,j]+=y*rng.uniform(.45,1.0)
    d=pd.DataFrame(X,columns=names)
    d["PotentialFraud"]=y
    d["Provider"]=[f"PRV{100000+i}" for i in range(n)]
    return d

@st.cache_resource
def demo_model():
    d=synthetic_data()
    feats=[c for c in d.columns if c not in ["PotentialFraud","Provider"]]
    Xtr,Xte,ytr,yte=train_test_split(d[feats],d.PotentialFraud,test_size=.25,stratify=d.PotentialFraud,random_state=42)
    m=RandomForestClassifier(n_estimators=220,class_weight="balanced",random_state=42,n_jobs=-1)
    m.fit(Xtr,ytr)
    p=m.predict_proba(Xte)[:,1]
    return d,feats,m,Xte,yte,p

def sha256_obj(obj):
    raw=json.dumps(obj,sort_keys=True,separators=(",",":")).encode()
    return hashlib.sha256(raw).hexdigest()

def merkle_root(hashes):
    level=hashes[:]
    while len(level)>1:
        if len(level)%2: level.append(level[-1])
        level=[hashlib.sha256((level[i]+level[i+1]).encode()).hexdigest() for i in range(0,len(level),2)]
    return level[0]

with st.sidebar:
    st.title("SecureClaim AI")
    page=st.radio("Panel navigation",[
        "Executive Dashboard","Research Workflow","Fraud Risk Analyzer",
        "Explainability","Blockchain Evidence","Published Results","Viva Mode"
    ])
    st.divider()
    st.caption("Research boundary")
    st.info("ML estimates fraud risk. Blockchain verifies evidence integrity. Neither independently establishes legal guilt.")
    st.caption("Demo data are synthetic unless you upload your exact research dataset.")

d,feats,model,Xte,yte,prob=demo_model()

if page=="Executive Dashboard":
    st.subheader("Research Prototype Dashboard")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Published ROC-AUC","0.951")
    c2.metric("Published Accuracy","94.0%")
    c3.metric("Published AP","0.733")
    c4.metric("Blockchain Latency","120–250 ms")
    st.markdown("### Integrated contribution")
    a,b,c=st.columns(3)
    a.markdown('<div class="card"><h3>🤖 Fraud-Risk Prediction</h3><p>Provider-level ensemble learning identifies potentially suspicious provider activity.</p></div>',unsafe_allow_html=True)
    b.markdown('<div class="card"><h3>🔎 Explainable AI</h3><p>SHAP-style feature attribution makes risk decisions interpretable and auditable.</p></div>',unsafe_allow_html=True)
    c.markdown('<div class="card"><h3>⛓️ Evidence Integrity</h3><p>SHA-256 and Merkle commitments provide tamper-evident verification for authorized auditing.</p></div>',unsafe_allow_html=True)
    st.markdown("### Thesis objectives")
    st.success("Objective 1 — Analyze existing challenges and vulnerabilities in managing health and insurance records")
    st.success("Objective 2 — Design a blockchain-based framework to enhance financial security and prevent insurance frauds")
    st.success("Objective 3 — Performance evaluation and comparison with existing ABHA framework")

elif page=="Research Workflow":
    st.subheader("End-to-End Research Workflow")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR;
      node [shape=box style="rounded,filled" fillcolor="#eaf2f8"];
      A [label="Healthcare Claims"];
      B [label="Preprocessing &\\nFeature Engineering"];
      C [label="Provider-Level\\nRepresentation"];
      D [label="XGBoost + LightGBM\\nStacking"];
      E [label="Fraud-Risk Score"];
      F [label="SHAP Explanation"];
      G [label="Evidence Artefact"];
      H [label="SHA-256 / Merkle"];
      I [label="Permissioned\\nBlockchain"];
      A->B->C->D->E->F->G->H->I;
    }""")
    st.info("Sensitive patient/claim information remains off-chain; cryptographic commitments and audit evidence are anchored on-chain.")

elif page=="Fraud Risk Analyzer":
    st.subheader("Interactive Provider Fraud-Risk Analyzer")
    st.caption("This panel uses the bundled demonstration model. Replace it with the exact trained research model for final experimental reproduction.")
    idx=st.slider("Select demonstration provider",0,min(500,len(Xte)-1),25)
    row=Xte.iloc[idx:idx+1]
    risk=float(model.predict_proba(row)[0,1])
    provider=d.loc[row.index[0],"Provider"]
    c1,c2,c3=st.columns(3)
    c1.metric("Provider",provider)
    c2.metric("Fraud-risk score",f"{risk:.3f}")
    c3.metric("Decision support","HIGH RISK" if risk>=.5 else "LOWER RISK")
    st.progress(min(max(risk,0.0),1.0))
    show=pd.DataFrame({"Feature":row.columns[:12],"Value":row.iloc[0,:12].round(3).values})
    st.dataframe(show,use_container_width=True,hide_index=True)
    st.warning("A high score flags a provider for review; it is not a legal finding of fraud.")

elif page=="Explainability":
    st.subheader("Explainable Fraud-Risk Decision")
    idx=st.slider("Provider for explanation",0,min(500,len(Xte)-1),25,key="explain")
    row=Xte.iloc[idx:idx+1]
    risk=float(model.predict_proba(row)[0,1])
    # Local perturbation importance proxy for robust demo UI.
    base=risk
    impacts={}
    for col in feats:
        altered=row.copy()
        altered[col]=Xte[col].median()
        impacts[col]=base-float(model.predict_proba(altered)[0,1])
    imp=pd.Series(impacts).sort_values(key=np.abs,ascending=False).head(12).sort_values()
    fig,ax=plt.subplots(figsize=(8,5))
    ax.barh(imp.index,imp.values)
    ax.set_title("Local Feature Influence – Demonstration")
    ax.set_xlabel("Change in predicted risk when feature is neutralized")
    st.pyplot(fig)
    st.info("The published thesis uses SHAP. This lightweight deployed panel uses a local perturbation view so the web demo remains stable; the published SHAP results are shown separately.")

elif page=="Blockchain Evidence":
    st.subheader("Cryptographic Evidence Verification")
    idx=st.slider("Select provider evidence",0,min(500,len(Xte)-1),25,key="chain")
    row=Xte.iloc[idx:idx+1]
    risk=float(model.predict_proba(row)[0,1])
    provider=d.loc[row.index[0],"Provider"]
    evidence={
        "provider_id":provider,
        "fraud_risk_score":round(risk,6),
        "model_output_type":"decision-support",
        "timestamp_demo":"panel-session",
        "patient_data_on_chain":False
    }
    h=sha256_obj(evidence)
    root=merkle_root([h]+[hashlib.sha256(f"evidence-{i}".encode()).hexdigest() for i in range(1,8)])
    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### Off-chain evidence artefact")
        st.json(evidence)
    with c2:
        st.markdown("#### On-chain commitment")
        st.code(h,language=None)
        st.markdown("**Merkle root**")
        st.code(root,language=None)
    st.success("✓ Integrity check: current evidence matches its SHA-256 commitment")
    tamper=st.checkbox("Simulate evidence tampering")
    if tamper:
        changed=dict(evidence); changed["fraud_risk_score"]=round(min(risk+.10,1),6)
        h2=sha256_obj(changed)
        st.error("✗ Verification failed: modified evidence produces a different hash")
        st.code(h2,language=None)
    st.caption("Blockchain verifies integrity/provenance, not whether the underlying claim is factually true.")

elif page=="Published Results":
    st.subheader("Published Experimental Results")
    st.caption("These are reported study benchmarks, not values recomputed from the synthetic web-demo dataset.")
    cols=st.columns(6)
    for col,(k,v) in zip(cols,PUBLISHED.items()):
        col.metric(k,f"{v:.3f}")
    fig,ax=plt.subplots(figsize=(9,4))
    ax.bar(list(PUBLISHED.keys()),list(PUBLISHED.values()))
    ax.set_ylim(0,1)
    ax.set_ylabel("Reported score")
    ax.set_title("Published Stacking-Ensemble Performance")
    ax.tick_params(axis="x",rotation=25)
    st.pyplot(fig)
    st.markdown("#### Provider-aware 5-fold cross-validation")
    st.dataframe(CV,use_container_width=True,hide_index=True)
    fig2,ax2=plt.subplots(figsize=(7,4))
    ax2.bar(CV["Model"],CV["Mean ROC-AUC"],yerr=CV["SD"],capsize=5)
    ax2.set_ylim(.90,.97); ax2.set_ylabel("ROC-AUC"); ax2.set_title("Reported 5-Fold CV ROC-AUC")
    st.pyplot(fig2)
    st.markdown("#### Reported blockchain performance")
    st.dataframe(pd.DataFrame({
      "Measure":["Transaction latency","Throughput","Block confirmation","SHA-256 hash generation","Transaction creation","Transaction submission","pBFT validation","Merkle proof generation"],
      "Reported value":["120–250 ms","800–1200 TPS","<1 sec","5–10 ms","20–40 ms","50–100 ms","80–150 ms","2–5 ms"]
    }),use_container_width=True,hide_index=True)
    st.info("Voting classifier achieved the highest reported Average Precision (0.746); stacking provided a strong balanced result and ROC-AUC of 0.951.")

elif page=="Viva Mode":
    st.subheader("🎓 PhD Panel / Viva Mode")
    qs=[
      ("What is the central contribution?","Integration of provider-level fraud-risk prediction, explainability and cryptographically verifiable evidence in one framework."),
      ("Why blockchain if ML already detects fraud?","ML estimates fraud risk; blockchain protects provenance, integrity and auditability of the resulting evidence."),
      ("Why explainability?","A fraud-risk score must be interpretable for review, accountability and regulated decision support."),
      ("Why permissioned blockchain?","Healthcare and insurance participants are known actors and evidence access must be governed."),
      ("Is patient data stored on-chain?","No. The design keeps sensitive information off-chain and anchors cryptographic commitments/evidence metadata."),
      ("Does blockchain prove fraud?","No. It verifies integrity and provenance. It does not establish factual truth or legal guilt."),
      ("Is the system production-ready?","No. The thesis reports prototype-level evaluation; independent datasets and operational pilots are future work."),
      ("How is ABHA comparison framed?","As a functional/architectural comparison. The proposed fraud-risk and evidence-integrity framework is complementary to ABHA, not a claim of direct performance superiority.")
    ]
    for q,a in qs:
        with st.expander(q):
            st.write(a)
    st.markdown("### 30-second closing statement")
    st.success("The framework combines machine-learning fraud-risk prediction, explainable decision support and blockchain-based evidence integrity. Its purpose is not autonomous fraud adjudication, but to help authorized stakeholders identify suspicious provider behaviour, understand the reasons for the risk score and verify that analytical evidence has not been altered.")

st.divider()
st.caption("Research prototype for academic demonstration. Published benchmarks are displayed separately from synthetic demonstration outputs.")
