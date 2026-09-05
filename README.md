
# SecureClaim AI — PhD Research Prototype

**Thesis:** A Blockchain-Based Framework to Prevent Insurance Frauds in the Healthcare Sector

This repository contains a Streamlit demonstration application for a PhD panel/viva. It presents the research workflow as an interactive software prototype:

Healthcare claims → provider-level features → fraud-risk prediction → explainability → evidence artefact → SHA-256/Merkle verification → permissioned-blockchain concept.

## Academic integrity note

The web app contains a **synthetic demonstration dataset** so the interface can run anywhere. It does **not** present demo outputs as the published experiment.

The **Published Results** page separately displays the verified results reported in the associated Scientific Reports study. For exact reproduction, replace the demo data/model with the original research dataset and trained pipeline.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy from GitHub to Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload all files from this folder to the repository root.
3. Sign in to Streamlit Community Cloud.
4. Create a new app and select your GitHub repository.
5. Set the entrypoint to `streamlit_app.py`.
6. Deploy.

## Panel flow

1. Executive Dashboard
2. Research Workflow
3. Fraud Risk Analyzer
4. Explainability
5. Blockchain Evidence
6. Published Results
7. Viva Mode

## Research boundary

- ML estimates fraud risk; it does not establish legal guilt.
- Blockchain verifies integrity/provenance; it does not prove factual truth.
- The system is a prototype-level decision-support framework.
