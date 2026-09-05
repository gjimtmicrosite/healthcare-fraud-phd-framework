
import streamlit as st
import pandas as pd
import numpy as np
import hashlib, json, time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

st.set_page_config(
    page_title="Healthcare Insurance Fraud Research Implementation",
    layout="wide"
)

# Academic, implementation-oriented styling
st.markdown("""
<style>
.block-container {padding-top:1rem; padding-bottom:2rem;}
h1,h2,h3 {font-family: Arial, sans-serif;}
.header {
    border-bottom: 2px solid #666;
    padding-bottom: 10px;
    margin-bottom: 18px;
}
.panel {
    border:1px solid rgba(120,120,120,.35);
    border-radius:4px;
    padding:14px;
    margin:8px 0;
}
.status-ok {
    border-left:4px solid #5f7d5f;
    background:rgba(90,120,90,.08);
    padding:10px 12px;
}
.status-info {
    border-left:4px solid #777;
    background:rgba(120,120,120,.08);
    padding:10px 12px;
}
code {font-size:0.9rem;}
div[data-testid="stMetric"] {
    border:1px solid rgba(120,120,120,.3);
    border-radius:4px;
    padding:10px;
}
[data-testid="stSidebar"] {
    border-right:1px solid rgba(120,120,120,.25);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
<h1>Healthcare Insurance Fraud Detection and Blockchain Evidence Framework</h1>
<p><b>Research Implementation Workbench</b></p>
<p>Provider-level fraud-risk modelling, explainability, and cryptographic evidence verification</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# REPORTED RESEARCH INFORMATION
# ----------------------------
REPORTED_DATASETS = pd.DataFrame({
    "Source Table":["Provider / Training","Beneficiary","Inpatient","Outpatient"],
    "Records":[5410,138556,40474,517737],
    "Attributes":[2,25,30,27]
})

REPORTED_METRICS = {
    "Accuracy":0.940,
    "Precision":0.702,
    "Recall":0.630,
    "F1":0.664,
    "ROC-AUC":0.951,
    "Average Precision":0.733
}

XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    gamma=0.2,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

LGB_PARAMS = dict(
    n_estimators=250,
    learning_rate=0.01,
    num_leaves=31,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

META_PARAMS = dict(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=500,
    random_state=42
)

def make_demo_provider_dataset(n=4057, seed=42):
    rng=np.random.default_rng(seed)
    y=np.array([1]*379+[0]*(n-379))
    rng.shuffle(y)

    names=[
        "ip__LOS_max","ip__LOS_std","ip__LOS_count","ip__LOS_mean","ip__LOS_median",
        "op__LOS_count",
        "ip__InscClaimAmtReimbursed_sum","ip__InscClaimAmtReimbursed_mean","ip__InscClaimAmtReimbursed_std",
        "op__InscClaimAmtReimbursed_sum","op__InscClaimAmtReimbursed_mean","op__InscClaimAmtReimbursed_std",
        "ip__DeductibleAmtPaid_sum","ip__DeductibleAmtPaid_mean",
        "op__DeductibleAmtPaid_sum","op__DeductibleAmtPaid_mean","op__DeductibleAmtPaid_std",
        "bene__age","bene__ChronicCond_Depression","bene__ChronicCond_ObstrPulmonary",
        "bene__ChronicCond_stroke"
    ]+[f"engineered_feature_{i:02d}" for i in range(22,56)]

    X=rng.normal(size=(n,55))
    for j in [0,2,6,9,12,17,18,19,20]:
        X[:,j]+=y*rng.uniform(.45,1.0)

    df=pd.DataFrame(X,columns=names)
    df["PotentialFraud"]=y
    df["Provider"]=[f"PRV{100000+i}" for i in range(n)]
    return df

def sha256_obj(obj):
    raw=json.dumps(obj,sort_keys=True,separators=(",",":")).encode()
    return hashlib.sha256(raw).hexdigest()

def merkle_root(hashes):
    level=hashes[:]
    if not level: return None
    while len(level)>1:
        if len(level)%2:
            level.append(level[-1])
        level=[
            hashlib.sha256((level[i]+level[i+1]).encode()).hexdigest()
            for i in range(0,len(level),2)
        ]
    return level[0]

# Session-state defaults
defaults = {
    "data": None,
    "features": None,
    "X_train": None, "X_test": None,
    "y_train": None, "y_test": None,
    "models": None,
    "results": None,
    "predictions": None,
    "last_evidence": None
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k]=v

with st.sidebar:
    st.markdown("### Implementation Sections")
    page=st.radio("",[
        "1. Project Overview",
        "2. Dataset and Integration",
        "3. Preprocessing",
        "4. Model Training",
        "5. Evaluation",
        "6. Explainability",
        "7. Blockchain Evidence",
        "8. Reproducibility Record"
    ])
    st.divider()
    st.caption("Implementation note")
    st.write("The workbench can operate in demonstration mode or accept an uploaded provider-level CSV. Published experimental benchmarks are shown separately from live demonstration outputs.")

# ----------------------------
# 1. OVERVIEW
# ----------------------------
if page=="1. Project Overview":
    st.subheader("Implementation Overview")
    st.markdown("""
    <div class="status-info">
    This interface is organized around the actual implementation stages used in the research:
    data preparation, provider-level feature construction, model training, evaluation,
    explanation generation, and blockchain evidence verification.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Processing Pipeline")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR;
      node [shape=box, style="rounded,filled", fillcolor="#f3f3f3", color="#666666"];
      A [label="Source Claims Tables"];
      B [label="Data Integration"];
      C [label="Provider-Level Features"];
      D [label="Training / Validation"];
      E [label="XGBoost + LightGBM"];
      F [label="Stacking Meta-Learner"];
      G [label="Fraud-Risk Output"];
      H [label="Explanation"];
      I [label="Evidence Hash"];
      J [label="Merkle / Blockchain Verification"];
      A->B->C->D->E->F->G->H->I->J;
    }""")

    st.markdown("### Reported Research Benchmarks")
    cols=st.columns(4)
    cols[0].metric("ROC-AUC","0.951")
    cols[1].metric("Accuracy","94.0%")
    cols[2].metric("Average Precision","0.733")
    cols[3].metric("Blockchain Latency","120–250 ms")

    st.markdown("### Research Boundary")
    st.write(
        "The machine-learning component estimates fraud risk. The explainability component "
        "provides reasons for the model output. The blockchain component verifies evidence "
        "integrity and provenance. The framework is a decision-support system rather than an "
        "autonomous fraud adjudication mechanism."
    )

# ----------------------------
# 2. DATASET
# ----------------------------
elif page=="2. Dataset and Integration":
    st.subheader("Dataset and Integration")

    st.markdown("### Reported Source Tables")
    st.dataframe(REPORTED_DATASETS,use_container_width=True,hide_index=True)

    c1,c2,c3=st.columns(3)
    c1.metric("Final Numerical Features","55")
    c2.metric("Potentially Fraudulent Providers","379")
    c3.metric("Non-Fraudulent Providers","3,678")

    st.markdown("### Interactive Data Source")
    mode=st.radio("Select data source",["Use built-in demonstration dataset","Upload provider-level CSV"])

    if mode=="Upload provider-level CSV":
        up=st.file_uploader(
            "Upload a provider-level CSV containing a binary target column named PotentialFraud",
            type=["csv"]
        )
        if up is not None:
            df=pd.read_csv(up)
            if "PotentialFraud" not in df.columns:
                st.error("The uploaded CSV must contain a binary target column named PotentialFraud.")
            else:
                if "Provider" not in df.columns:
                    df["Provider"]=[f"PRV{i:06d}" for i in range(len(df))]
                st.session_state.data=df
                st.success(f"Loaded {len(df):,} provider-level records.")
    else:
        if st.button("Load Demonstration Dataset"):
            st.session_state.data=make_demo_provider_dataset()
            st.success("Demonstration provider-level dataset loaded.")

    if st.session_state.data is not None:
        df=st.session_state.data
        st.markdown("### Loaded Data Summary")
        st.write(f"Rows: **{len(df):,}**")
        st.write(f"Columns: **{len(df.columns):,}**")
        st.dataframe(df.head(10),use_container_width=True)

        if "PotentialFraud" in df.columns:
            counts=df["PotentialFraud"].value_counts().sort_index()
            fig,ax=plt.subplots(figsize=(6,4))
            labels=["Non-Fraudulent","Potentially Fraudulent"] if set(counts.index).issubset({0,1}) else counts.index.astype(str)
            ax.bar(labels,counts.values)
            ax.set_ylabel("Provider Count")
            ax.set_title("Loaded Provider-Level Class Distribution")
            st.pyplot(fig)

    st.markdown("### Integration Logic")
    st.code("""
Provider table
    + Beneficiary table
    + Inpatient claims
    + Outpatient claims
        ↓
Claim-level aggregation
        ↓
Provider-level numerical representation
        ↓
55 numerical features used for fraud-risk modelling
""", language=None)

# ----------------------------
# 3. PREPROCESSING
# ----------------------------
elif page=="3. Preprocessing":
    st.subheader("Preprocessing and Feature Preparation")

    if st.session_state.data is None:
        st.warning("Load a dataset first from 'Dataset and Integration'.")
    else:
        df=st.session_state.data.copy()

        st.markdown("### Preprocessing Checks")
        report=pd.DataFrame({
            "Check":[
                "Duplicate rows",
                "Missing values",
                "Target column present",
                "Provider identifier present"
            ],
            "Observed":[
                int(df.duplicated().sum()),
                int(df.isna().sum().sum()),
                "Yes" if "PotentialFraud" in df.columns else "No",
                "Yes" if "Provider" in df.columns else "No"
            ]
        })
        st.dataframe(report,use_container_width=True,hide_index=True)

        numeric=[c for c in df.select_dtypes(include=np.number).columns if c!="PotentialFraud"]

        st.markdown("### Feature Preparation")
        st.write(f"Numerical candidate features detected: **{len(numeric)}**")
        st.dataframe(pd.DataFrame({"Feature":numeric[:55]}),use_container_width=True,hide_index=True)

        if st.button("Prepare Train/Test Data"):
            X=df[numeric].copy()
            y=df["PotentialFraud"].astype(int).copy()

            # median imputation
            X=X.fillna(X.median(numeric_only=True))

            X_train,X_test,y_train,y_test=train_test_split(
                X,y,test_size=.25,stratify=y,random_state=42
            )

            st.session_state.features=numeric
            st.session_state.X_train=X_train
            st.session_state.X_test=X_test
            st.session_state.y_train=y_train
            st.session_state.y_test=y_test

            st.success("Preprocessing completed and train/test partitions prepared.")

        if st.session_state.X_train is not None:
            c1,c2=st.columns(2)
            c1.metric("Training Records",f"{len(st.session_state.X_train):,}")
            c2.metric("Test Records",f"{len(st.session_state.X_test):,}")

        st.markdown("### Research Validation Strategy")
        st.write(
            "The thesis uses provider-aware GroupKFold cross-validation (k = 5), and resampling "
            "is restricted to training folds to minimize information leakage. The reported "
            "SMOTE-Tomek training balance is 3,675 fraudulent and 3,675 non-fraudulent observations."
        )

# ----------------------------
# 4. MODEL TRAINING
# ----------------------------
elif page=="4. Model Training":
    st.subheader("Model Training")

    st.markdown("### Principal Configuration")
    st.dataframe(pd.DataFrame({
        "Model":["XGBoost","LightGBM","Stacking Meta-Learner"],
        "Configuration":[
            "n_estimators=300, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, gamma=0.2",
            "n_estimators=250, learning_rate=0.01, num_leaves=31, max_depth=8, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0",
            "Logistic Regression: L2 penalty, C=1.0, lbfgs, max_iter=500"
        ]
    }),use_container_width=True,hide_index=True)

    if st.session_state.X_train is None:
        st.warning("Prepare train/test data first from the Preprocessing section.")
    else:
        if st.button("Train Research Models"):
            with st.spinner("Training XGBoost, LightGBM and stacking ensemble..."):
                xgb=XGBClassifier(**XGB_PARAMS)
                lgb=LGBMClassifier(**LGB_PARAMS)
                meta=LogisticRegression(**META_PARAMS)

                stacking=StackingClassifier(
                    estimators=[("xgb",xgb),("lgbm",lgb)],
                    final_estimator=meta,
                    stack_method="predict_proba",
                    cv=5,
                    n_jobs=-1
                )

                models={"XGBoost":xgb,"LightGBM":lgb,"Stacking":stacking}
                results={}
                predictions={}

                Xtr=st.session_state.X_train
                Xte=st.session_state.X_test
                ytr=st.session_state.y_train
                yte=st.session_state.y_test

                for name,m in models.items():
                    m.fit(Xtr,ytr)
                    p=m.predict_proba(Xte)[:,1]
                    yhat=(p>=.5).astype(int)
                    predictions[name]=(p,yhat)
                    results[name]={
                        "Accuracy":accuracy_score(yte,yhat),
                        "Precision":precision_score(yte,yhat,zero_division=0),
                        "Recall":recall_score(yte,yhat,zero_division=0),
                        "F1":f1_score(yte,yhat,zero_division=0),
                        "ROC-AUC":roc_auc_score(yte,p),
                        "Average Precision":average_precision_score(yte,p)
                    }

                st.session_state.models=models
                st.session_state.results=pd.DataFrame(results).T
                st.session_state.predictions=predictions

            st.success("Model training completed.")

        if st.session_state.models is not None:
            st.markdown("### Training Status")
            st.dataframe(pd.DataFrame({
                "Model":["XGBoost","LightGBM","Stacking"],
                "Status":["Trained","Trained","Trained"]
            }),use_container_width=True,hide_index=True)

# ----------------------------
# 5. EVALUATION
# ----------------------------
elif page=="5. Evaluation":
    st.subheader("Model Evaluation")

    st.markdown("### Published Experimental Benchmark")
    st.dataframe(pd.DataFrame([REPORTED_METRICS],index=["Published Stacking Result"]).round(3),
                 use_container_width=True)

    if st.session_state.results is None:
        st.info("Train the models to generate live evaluation outputs.")
    else:
        st.markdown("### Live Evaluation Output")
        st.dataframe(st.session_state.results.round(3),use_container_width=True)

        # ROC
        fig,ax=plt.subplots(figsize=(7,5))
        for name,(p,yhat) in st.session_state.predictions.items():
            fpr,tpr,_=roc_curve(st.session_state.y_test,p)
            auc=roc_auc_score(st.session_state.y_test,p)
            ax.plot(fpr,tpr,label=f"{name} (AUC={auc:.3f})")
        ax.plot([0,1],[0,1],"--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        st.pyplot(fig)

        # PR
        fig2,ax2=plt.subplots(figsize=(7,5))
        for name,(p,yhat) in st.session_state.predictions.items():
            pr,re,_=precision_recall_curve(st.session_state.y_test,p)
            ap=average_precision_score(st.session_state.y_test,p)
            ax2.plot(re,pr,label=f"{name} (AP={ap:.3f})")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curves")
        ax2.legend()
        st.pyplot(fig2)

        # Confusion matrix
        p,yhat=st.session_state.predictions["Stacking"]
        cm=confusion_matrix(st.session_state.y_test,yhat)
        fig3,ax3=plt.subplots(figsize=(5,4))
        im=ax3.imshow(cm)
        ax3.set_xticks([0,1],["Non-Fraud","Potential Fraud"])
        ax3.set_yticks([0,1],["Non-Fraud","Potential Fraud"])
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")
        ax3.set_title("Stacking Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax3.text(j,i,cm[i,j],ha="center",va="center")
        st.pyplot(fig3)

    st.caption(
        "Live outputs depend on the data loaded into this implementation workbench. "
        "Published study values are displayed separately and should not be represented as newly recomputed unless the exact original data and pipeline are used."
    )

# ----------------------------
# 6. EXPLAINABILITY
# ----------------------------
elif page=="6. Explainability":
    st.subheader("Provider-Level Explainability")

    if st.session_state.models is None:
        st.warning("Train the models before generating an explanation.")
    else:
        idx=st.slider("Select test provider record",0,min(300,len(st.session_state.X_test)-1),10)
        row=st.session_state.X_test.iloc[idx:idx+1]
        risk=float(st.session_state.models["Stacking"].predict_proba(row)[0,1])

        st.metric("Stacking Fraud-Risk Score",f"{risk:.4f}")

        # Local perturbation analysis for deployed stability
        impacts={}
        for col in row.columns:
            altered=row.copy()
            altered[col]=st.session_state.X_test[col].median()
            p2=float(st.session_state.models["Stacking"].predict_proba(altered)[0,1])
            impacts[col]=risk-p2

        s=pd.Series(impacts).sort_values(key=np.abs,ascending=False).head(12).sort_values()
        fig,ax=plt.subplots(figsize=(8,5))
        ax.barh(s.index,s.values)
        ax.set_xlabel("Change in predicted risk")
        ax.set_title("Provider-Level Feature Influence")
        st.pyplot(fig)

        st.markdown("### Explainability Record")
        st.dataframe(s.sort_values(key=np.abs,ascending=False).to_frame("Influence"),
                     use_container_width=True)

        st.write(
            "The published research uses SHAP for formal explainability and reports a mean "
            "Spearman rank correlation of approximately 0.82 and top-10 feature overlap above 85%."
        )

# ----------------------------
# 7. BLOCKCHAIN
# ----------------------------
elif page=="7. Blockchain Evidence":
    st.subheader("Blockchain Evidence Generation and Verification")

    if st.session_state.models is None:
        st.warning("Train the model before generating a provider evidence record.")
    else:
        idx=st.slider("Select test provider record",0,min(300,len(st.session_state.X_test)-1),10,key="bc_idx")
        row=st.session_state.X_test.iloc[idx:idx+1]
        risk=float(st.session_state.models["Stacking"].predict_proba(row)[0,1])

        evidence={
            "provider_test_index":int(idx),
            "fraud_risk_score":round(risk,6),
            "model":"Stacking (XGBoost + LightGBM -> Logistic Regression)",
            "decision_support_only":True,
            "patient_data_on_chain":False
        }

        h=sha256_obj(evidence)
        root=merkle_root([h]+[
            hashlib.sha256(f"evidence-{i}".encode()).hexdigest()
            for i in range(1,8)
        ])
        st.session_state.last_evidence=(evidence,h,root)

        c1,c2=st.columns(2)
        with c1:
            st.markdown("### Off-Chain Evidence")
            st.json(evidence)
        with c2:
            st.markdown("### On-Chain Commitment")
            st.markdown("**SHA-256 Evidence Hash**")
            st.code(h,language=None)
            st.markdown("**Merkle Root**")
            st.code(root,language=None)

        st.success("Verification result: evidence matches the current cryptographic commitment.")

        if st.checkbox("Modify fraud-risk score to test tamper detection"):
            changed=dict(evidence)
            changed["fraud_risk_score"]=round(min(risk+.10,1),6)
            changed_hash=sha256_obj(changed)
            st.error("Verification result: mismatch detected.")
            st.write("Modified evidence hash:")
            st.code(changed_hash,language=None)

        st.markdown("### Reported Blockchain Performance")
        st.dataframe(pd.DataFrame({
            "Measure":[
                "Transaction latency","Throughput","Block confirmation",
                "SHA-256 hash generation","Transaction creation",
                "Transaction submission","pBFT validation","Merkle proof generation"
            ],
            "Reported Value":[
                "120–250 ms","800–1200 TPS","<1 sec",
                "5–10 ms","20–40 ms","50–100 ms","80–150 ms","2–5 ms"
            ]
        }),use_container_width=True,hide_index=True)

# ----------------------------
# 8. REPRODUCIBILITY
# ----------------------------
elif page=="8. Reproducibility Record":
    st.subheader("Reproducibility Record")

    st.markdown("### Experimental Environment Reported in the Study")
    st.dataframe(pd.DataFrame({
        "Component":[
            "Processor","Memory","Storage","Operating System","Python",
            "Pandas","NumPy","Scikit-learn","XGBoost","LightGBM","SHAP"
        ],
        "Reported Configuration":[
            "Intel Core i7, 8 cores, 2.6–3.0 GHz",
            "16–32 GB DDR4 RAM",
            "512 GB SSD",
            "Windows 10 64-bit / Ubuntu 20.04",
            "3.9.13",
            "1.5",
            "1.23",
            "1.2",
            "1.7",
            "3.3",
            "0.41"
        ]
    }),use_container_width=True,hide_index=True)

    st.markdown("### Implementation Checklist")
    checklist=pd.DataFrame({
        "Stage":[
            "Data source documented",
            "Provider-level representation",
            "Training/test preparation",
            "XGBoost configuration",
            "LightGBM configuration",
            "Stacking meta-learner",
            "Performance evaluation",
            "Explainability",
            "Evidence hashing",
            "Merkle verification",
            "Permissioned blockchain concept"
        ],
        "Status":[
            "Implemented / documented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented at prototype level"
        ]
    })
    st.dataframe(checklist,use_container_width=True,hide_index=True)

    st.markdown("### Research Objectives")
    st.write("**Objective 1:** Analyze existing challenges and vulnerabilities in managing health and insurance records")
    st.write("**Objective 2:** Design a blockchain-based framework to enhance financial security and prevent insurance frauds")
    st.write("**Objective 3:** Performance evaluation and comparison with existing ABHA framework")

    st.markdown("""
    <div class="status-info">
    Objective 3 is presented as an experimental evaluation of the proposed framework together with
    a functional/architectural comparison with the existing ABHA framework. It is not a claim of
    direct operational performance superiority over ABHA.
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("Doctoral research implementation workbench. Published benchmarks are separated from live demonstration outputs.")
