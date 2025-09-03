# PN-Usecase
Multi-Checkpoint Delivery Delay Prediction
## Abstract 

Develop a predictive model that predicts parcel level delays and container level delays. At every checkpoint where parcel is scanned. Based on historical events we need to predict the duration of delivery for the next level.

<img width="1021" height="200" alt="architecture" src="https://github.com/user-attachments/assets/9b59db60-abc8-448a-aa07-96553bf081d5" />

Idea : 
The core problem involves developing a multi-level predictive system that forecasts delivery delays at both parcel and container levels across various checkpoints in the logistics pipeline. The system needs to predict delivery durations at each scanning checkpoint, from customer drop-off through node hubs, sorting terminals, distribution terminals, and final delivery to consumers

The two granularities:
Parcel-level predictions: Estimating delivery duration for individual packages at each scanning checkpoint
Container-level predictions: Predicting delays for containers (which hold multiple parcels)

The system needs to predict the "duration to next checkpoint" dynamically as items move through the logistics network, updating predictions at each scan point based on:

- Historical transit patterns
- Current network conditions
- Package/container characteristics
- Route information

## Current Work 
Existing Approaches and Methodologies
paper - robot but can borrow maths [https://www.mdpi.com/2227-7390/12/20/3201]

XGBoost and CatBoost models: 71.5% to 99.9% ROC-AUC scores 
[https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5054862]- read paper

Neural Network and SVM approaches: 77% prediction accuracy for container shipping delays
[paper - https://scholarspace.manoa.hawaii.edu/server/api/core/bitstreams/9b9631b4-7422-4bae-929a-7258bf8652e9/content]

Random Forest implementations: 97.4% accuracy in supply chain delay prediction [Late Delivery Risk Prediction - https://github.com/PolinaBurova/Predicting-Delivery-Delays-in-Supply-Chain]

## ------------------------
Key Technologies in Use:

Ensemble methods: Random Forest, Gradient Boosting, XGBoost dominating the field
[Predicting Delays In Delivery Process Using Machine Learning-Based Approach - https://hammer.purdue.edu/articles/thesis/Predicting_Delays_In_Delivery_Process_Using_Machine_Learning-Based_Approach/13350764?file=25732868]

Deep learning approaches: LSTM networks for sequential prediction

Hybrid models: Combining multiple algorithms for improved performance

[https://arxiv.org/pdf/2105.08526 - Massively parallel real-time predictions of train delay propagation - parallel real time prediction ]
[Leveraging LSTM for Accurate ETA Classification and Delay Prediction in College Bus Tracking Systems - https://ieeexplore.ieee.org/document/11081054]
[https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11081054]


OSS datasets : 

- https://www.kaggle.com/datasets/nicolemachado/transportation-and-logistics-tracking-dataset/data
- https://pubsonline.informs.org/doi/10.1287/trsc.2022.1173
- https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis/data?select=tokenized_access_logs.csv
