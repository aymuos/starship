# 📚 M.Tech Thesis Paper Tracker

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

---

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

### A machine learning approach for enhancing supply chain visibility with graph-based learning
**Domain**: Supply Chain / Logistics  
**ML Area**: Graph Convolutional Networks  

**Authors**: Mohammadreza Nazari, Afshin Oroojlooy, Lawrence Snyder, Martin Takáč  
**Published**: NeurIPS Workshop, 2018  

**Abstract (Summary)**:  
In today’s globalised trade, supply chains form complex networks spanning multiple organisations and even countries, making them highly vulnerable to disruptions. These vulnerabilities, highlighted by recent global crises, underscore the urgent need for improved visibility and resilience of the supply chain. However, data-sharing limitations often hinder the achievement of comprehensive visibility between organisations or countries due to privacy, security, and regulatory concerns. Moreover, most existing research studies focused on individual firm- or product-level networks, overlooking the multifaceted interactions among diverse entities that characterise real-world supply chains, thus limiting a holistic understanding of supply chain dynamics. To address these challenges, we propose a novel approach that integrates Federated Learning (FL) and Graph Convolutional Neural Networks (GCNs) to enhance supply chain visibility through relationship prediction in supply chain knowledge graphs. FL enables collaborative model training across countries by facilitating information sharing without requiring raw data exchange, ensuring compliance with privacy regulations and maintaining data security. GCNs empower the framework to capture intricate relational patterns within knowledge graphs, enabling accurate link prediction to uncover hidden connections and provide comprehensive insights into supply chain networks. Experimental results validate the effectiveness of the proposed approach, demonstrating its ability to accurately predict relationships within country-level supply chain knowledge graphs. This enhanced visibility supports actionable insights, facilitates proactive risk management, and contributes to the development of resilient and adaptive supply chain strategies, ensuring that supply chains are better equipped to navigate the complexities of the global economy.

**Why it’s interesting**:  
- Global supply chains are increasingly vulnerable due to geopolitical tensions, pandemics, and climate events — making resilience and visibility a critical challenge.

- This paper tackles the data-sharing dilemma across organizational and national boundaries by using Federated Learning, which is highly relevant in regulated, privacy-sensitive environments.

- Unlike prior work focusing on individual firm-level networks, this approach leverages Graph Convolutional Networks on country-level supply chain knowledge graphs, enabling a holistic understanding of interdependencies across the supply network.

- The combination of FL + GCN is novel and practically viable, showing potential to drive real-time risk detection, hidden link discovery, and policy-compliant collaboration.

- It opens up future research opportunities in areas like cross-border inventory coordination, resilient logistics planning, or decentralized anomaly detection in global trade.



## 🔍 Topic Categorization

**Techniques:**
- Forecasting
- Deep Learning
- Reinforcement Learning
- Anomaly Detection
- Optimization
- Graph-based ML

**Applications:**
- Demand Forecasting  
- Inventory Optimization  
- Delivery Route Planning  
- Production Scheduling  
- Quality Assurance  
- Predictive Maintenance  

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
