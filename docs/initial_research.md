# 📚 M.Tech Thesis Paper Tracker: ML in Retail, Supply Chain & Manufacturing

This repository contains a curated list of research papers exploring the application of **machine learning (ML)** in:

- 🛒 Retail  
- 🚚 Supply Chain  
- 🏭 Manufacturing  

The objective is to understand current research trends across various ML domains and identify a compelling thesis topic with both academic and industrial relevance.

---

## 🗂️ Entry Format

Each paper entry includes:

- **Domain**: Application area
- **ML Area**: Technique or subfield of ML
- **Authors**: Authors of the paper
- **Published**: Venue and year
- **Abstract (Summary)**: A 3–5 line summary of the contribution
- **Why it’s interesting**: Notes on relevance, novelty, and future scope

---

## ✅ Tracked Papers

### [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)

**Domain**: Retail  
**ML Area**: Probabilistic Forecasting / Deep Learning / Time Series  

**Authors**: David Salinas, Valentin Flunkert, Jan Gasthaus, Tim Januschowski  
**Published**: Amazon Research, 2017  

**Abstract (Summary)**:  
DeepAR is a deep learning method for forecasting multivariate time series using autoregressive RNNs. Unlike classical models, it produces probabilistic forecasts and handles thousands of related time series — like those in retail demand.

**Why it’s interesting**:  
- Retail datasets often include large numbers of similar time series (e.g., product-level sales)  
- Handles cold-start scenarios using item metadata  
- Can be extended for hierarchical forecasting or multimodal inputs

---

### [A Deep Reinforcement Learning Framework for the Vehicle Routing Problem](https://arxiv.org/abs/1802.04240)

**Domain**: Supply Chain / Logistics  
**ML Area**: Reinforcement Learning / Combinatorial Optimization  

**Authors**: Mohammadreza Nazari, Afshin Oroojlooy, Lawrence Snyder, Martin Takáč  
**Published**: NeurIPS Workshop, 2018  

**Abstract (Summary)**:  
This work introduces an attention-based deep reinforcement learning model to solve the capacitated vehicle routing problem (CVRP). It avoids hand-crafted heuristics and learns policies through simulation.

**Why it’s interesting**:  
- Traditional logistics problems are NP-hard and rely on heuristics  
- Learns routing decisions from data, adaptable to real-time needs  
- Can inspire work on multi-agent routing, delivery drones, or joint inventory-routing problems

---

### [On the use of machine learning in supply chain management: a systematic review](https://www.sciencedirect.com/science/article/pii/S1568494625003539)

**Domain**: Supply Chain  
**ML Area**: Systematic Review / Forecasting / Reinforcement Learning  

**Authors**: Babai et al.  
**Published**: IMA Journal of Management Mathematics, 2025  

**Abstract (Summary)**:  
This extensive review classifies hundreds of ML studies in supply chain management (SCM), especially focusing on demand forecasting, inventory management, distribution, and transportation. Supervised learning dominates, but reinforcement learning is gaining traction.

**Why it’s interesting**:  
- Provides a panoramic view of ML use cases in SCM  
- Highlights evolution trends and future directions  
- Helps identify research gaps in ML applications beyond demand forecasting

---

### Machine Learning in Supply Chain Management: A Systematic Literature Review

**Domain**: Supply Chain  
**ML Area**: Predictive Analytics / Decision Support  

**Authors**: Not listed  
**Published**: IJSOM  

**Abstract (Summary)**:  
This systematic review covers the predictive capabilities of ML across inventory levels, supplier quality, procurement, and logistics. It highlights ML’s most influential impact in demand forecasting and decision support.

**Why it’s interesting**:  
- Broadens perspective beyond forecasting to procurement and supplier quality  
- Strong basis for identifying ML methods suitable for each supply chain component

---

### Enhancing Supply Chain Management with Deep Learning and ML

**Domain**: Supply Chain  
**ML Area**: Deep Learning / Decision Support / Optimization  

**Abstract (Summary)**:  
Reviews the role of deep learning and ML in supplier selection, production, inventory control, and other core aspects of the supply chain.

**Why it’s interesting**:  
- Useful overview of DL integration into traditional SCM tasks  
- Identifies multi-stage optimization opportunities

---

### An Innovative Machine Learning Model for Supply Chain Management

**Domain**: Supply Chain  
**ML Area**: Conditional GANs / Optimization  

**Abstract (Summary)**:  
Introduces a CGAN-based approach for dynamic supply chain member selection.

**Why it’s interesting**:  
- Uses generative models for decision-making tasks  
- Potentially applicable to resilient network design

---

### Kim et al. (2024): ML in Biodiesel Supply Chain

**Domain**: Supply Chain / Energy  
**ML Area**: Random Forests / Neural Networks / Predictive Modeling  

**Abstract (Summary)**:  
Finds Random Forests and Artificial Neural Networks most accurate for predicting feedstock yield and productivity in complex logistics environments.

**Why it’s interesting**:  
- Focuses on sustainable supply chains  
- Tackles real-world yield prediction under uncertainty

---

## 🛒 Retail

### [Identifying Application Areas for Machine Learning in the Retail Sector (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10245364/)

**Domain**: Retail  
**ML Area**: Supervised Learning / Decision Support  

**Abstract (Summary)**:  
Systematically identifies 21 key ML application areas in online and offline retail, emphasizing economic decision-support (e.g., marketing, inventory optimization, recommendations).

**Why it’s interesting**:  
- Holistic mapping of ML across retail value chain  
- Useful taxonomy to select focus area

---

### Machine Learning in Retail: Practical Use Cases and Impact (Akkio, 2024)

**Domain**: Retail  
**ML Area**: Real-World Systems / Personalization / Forecasting  

**Abstract (Summary)**:  
Highlights Amazon (recommendation systems, Amazon Go) and H&M (store layout and demand prediction) as real-world leaders in ML-driven retail innovation.

**Why it’s interesting**:  
- Shows maturity of ML in retail deployment  
- Connects academic ideas with production systems

---

## 🏭 Manufacturing

### [Machine Learning and Deep Learning Based Predictive Quality in Manufacturing](https://www.researchgate.net/journal/Journal-of-Intelligent-Manufacturing-1572-8145/publication/360922608_Machine_learning_and_deep_learning_based_predictive_quality_in_manufacturing_a_systematic_review/links/6292d60088c32b037b5a607d/Machine-learning-and-deep-learning-based-predictive-quality-in-manufacturing-a-systematic-review.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uRG93bmxvYWQiLCJwcmV2aW91c1BhZ2UiOiJwdWJsaWNhdGlvbiJ9fQ&__cf_chl_tk=.5JGhjOqRW7yBIFxBcrTCmuCP_Idz9ZoEYOOCwJ8G70-1754407531-1.0.1.1-nbX2P7_Ga3EUXsal1F9jFiIBkF4KO6nSXAZvuuZQlZw)

**Domain**: Manufacturing  
**ML Area**: Predictive Quality / Sensor Data / Inspection  

**Abstract (Summary)**:  
Systematic review from 2012–2021 categorizing predictive quality approaches—sensor monitoring to automated defect detection—using ML/DL.

**Why it’s interesting**:  
- Maps quality control evolution  
- Identifies gaps in integration and standardization

---

### Machine Learning Applications in Production Lines: A Systematic Review

**Domain**: Manufacturing  
**ML Area**: Predictive Maintenance / Optimization / Quality Control  

**Abstract (Summary)**:  
Synthesizes studies of ML along production lines, emphasizing impact in predictive maintenance, yield optimization, and quality improvement.

**Why it’s interesting**:  
- Valuable for identifying high-impact areas in smart manufacturing  
- Good base for building real-time production ML models

---

### Industrial Use Cases Overview (MobiDev, 2025)

**Domain**: Manufacturing  
**ML Area**: NLP / Document Processing / LLM Integration  

**Abstract (Summary)**:  
Describes applications like intelligent document processing, NLP for contract management, and integration of LLMs with manufacturing workflows.

**Why it’s interesting**:  
- Future-focused on LLM + ML integration  
- Useful for thesis ideas in industrial NLP

---

## 🔎 General ML Papers Impacting All Sectors

### 21 Most Cited Machine Learning Papers (Doradolist)

**Domain**: General ML  
**ML Area**: Foundational Algorithms  

**Abstract (Summary)**:  
Lists landmark ML papers (e.g., ResNet, Transformers) widely cited in applied domains like supply chain, retail, and manufacturing.

**Why it’s interesting**:  
- Basis for most modern ML pipelines  
- Essential background for implementing advanced models

---

## 🏛️ Highly Cited Journals

- **Journal of Intelligent Manufacturing**: Top venue for predictive maintenance, quality assurance, and intelligent automation research.

---

## 📌 Goals

- Explore research trends in each domain  
- Understand real-world problems and modeling approaches  
- Identify a high-impact, feasible M.Tech thesis topic  

---

## 📎 Useful Resources

- [Awesome Supply Chain Optimization](https://github.com/daoleno/awesome-supply-chain-optimization)  
- [M5 Forecasting Dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy)  
- [Papers With Code – Manufacturing](https://paperswithcode.com/task/manufacturing)

---
