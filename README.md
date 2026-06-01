# Credit Scoring Business Understanding

## 1. Basel II and Model Interpretability

The Basel II Accord emphasizes that financial institutions must maintain transparent, well-documented, and interpretable credit risk models. This is critical because lending decisions must be explainable to regulators, auditors, and internal risk teams. As a result, even when high-performing machine learning models are used, their outputs must remain traceable and justifiable. This influences the preference for simpler models such as Logistic Regression in traditional credit scoring systems.

---

## 2. Proxy Variable for Default

In this project, there is no direct label indicating customer default behavior. Therefore, a proxy target variable is constructed using behavioral patterns such as Recency, Frequency, and Monetary (RFM) features.

Customers with low engagement and low transaction value are assumed to have higher risk of default.

However, this introduces risks:

* The proxy may not perfectly represent real default behavior
* Business assumptions can introduce bias
* Model performance depends heavily on proxy quality

Thus, the model is an approximation rather than a ground-truth credit risk system.

---

## 3. Interpretability vs Performance Trade-off

There is a key trade-off in credit risk modeling:

* Logistic Regression + WoE:

  * Highly interpretable
  * Regulatory friendly
  * Easier to explain to stakeholders
  * Lower predictive power

* Tree-based models (Random Forest, Gradient Boosting):

  * Higher predictive accuracy
  * Capture nonlinear relationships
  * Less interpretable
  * Harder to justify under regulatory constraints

In regulated environments like banking, interpretability often takes priority over raw predictive performance.
