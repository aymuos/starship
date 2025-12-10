LADE DATASET 

- LaDe is a large-scale public last‑mile delivery dataset with 10.677M packages handled by 21k couriers over 6 months across five Chinese cities, split into two subsets—LaDe‑P for pickups and LaDe‑D for deliveries
  —with rich event timestamps and GPS to enable phase‑level delay analysis and benchmarking tasks like route prediction and ETA forecasting. 

- Dataset : https://huggingface.co/datasets/Cainiao-AI/LaDe

- designed to support research in logistics, spatio‑temporal learning (data having both space and time relations), and operations, featuring both package pickup and delivery scenarios with comprehensive package, stop, courier, and task‑event information

- It contains 10.677 million package records, 619k trajectories, and 16.755 million GPS pings from 21k couriers across 6 months, sampled from five diverse cities to capture different spatio‑temporal patterns.

- The two sub‑datasets are LaDe‑P (pickup) and LaDe‑D (delivery), each presented as per‑package CSV tables with event timestamps (accept, pickup or delivery) and nearest-event GPS fixes

- Subsets and fields
    > LaDe‑P (pickup) includes package_id, time_window_start/end, stop coordinates (lat/lng), city, region_id, aoi_id, aoi_type, courier_id, accept_time with nearest accept_gps_time and coordinates, and pickup_time with nearest pickup_gps_time and coordinates, plus a date field ds.

    > LaDe‑D (delivery) includes package_id, stop coordinates, city, region_id, aoi_id, aoi_type, courier_id, accept_time with nearest accept_gps fix, and delivery_time with nearest delivery_gps fix, plus ds, enabling accept→delivery phase timing and SLA calculations when time windows apply.

- Tasks and benchmarks
> Route prediction (LaDe‑P): given unfinished tasks for a courier at time t, predict the future service order; baseline families include TimeGreedy/DistanceGreedy, OR‑Tools/LightGBM, and deep models like DeepRoute, FDNET, and Graph2Route, evaluated with HR@k, Kendall‑tau (KRC), local sequence distance (LSD), and edit distance (ED).

 > ETA prediction (LaDe‑D): given a query time and unfinished task set, predict per‑task remaining time; baselines include SPEED, KNN, LightGBM, MLP, FDNET, and RankETPA, reported with MAE, RMSE, and ACC@20/30 by city to highlight cross‑city generalization gaps.

> Spatio‑temporal graph forecasting (LaDe‑P): node=region, signal=hourly pickup counts; deep STG models like DCRNN, STGCN, GWNET, ASTGCN, MTGNN, AGCRN, and STGNCDE outperform historical average, illustrating demand forecasting value for control and planning

----------------------------------------------------------------------------------------------------------

Short answer: To generalize models built on LaDe across cities, seasons, and datasets, combine cross‑city transfer learning, domain generalization, self‑supervised trajectory pretraining, meta‑learning, test‑time adaptation, and calibrated uncertainty estimation, aligning with LaDe’s city‑wise splits and spatiotemporal structure.[1][2][3]

### Cross‑city adaptation
- Use domain adaptation for spatiotemporal tasks so models trained on one city adapt to another with limited labels, leveraging adversarial spatial alignment and temporal attentive adaptation like STAN and related cross‑city frameworks.[4][1]
- Earlier methods such as RegionTrans show effective cross‑city transfer for deep spatiotemporal prediction, while newer cross‑city approaches (e.g., D2MHyper) target higher‑order spatial relations to improve robustness under geographic shift.[5][6]

### Domain generalization
- Train with multiple source cities and minimize domain gaps using invariant learning or discrepancy matching (e.g., IRM, CORAL, MMD, DANN) to learn city‑invariant features without target data.[2][7]
- Recent DG studies caution that ERM with carefully designed training and domain‑aware weighting can rival or beat some invariant learners, so benchmark IRM/CORAL/MMD/DANN against strong ERM and GroupDRO variants.[8][2]

### Self‑supervised pretraining
- Pretrain encoders on trajectories and event sequences with contrastive time‑series methods (e.g., TS2Vec, TF‑C) to learn transferable representations that fine‑tune well on new cities with few labels.[9][10][11]
- Structure‑preserving contrastive objectives for spatial time series can further stabilize transfer by respecting graph and temporal locality during pretraining.[12]

### Meta‑learning
- Apply spatiotemporal meta‑learning (MetaST, ST‑MetaNet+) to learn initializations that adapt rapidly to new cities, AOIs, or seasons using few examples and dynamic graph/temporal context.[13][14]
- Meta‑learn city‑conditioned adapters for graph and recurrent layers so per‑city peculiarities are captured while sharing generalizable structure.[13]

### Test‑time adaptation
- Equip models with source‑free test‑time training to handle abrupt distribution shifts (e.g., storms or festivals) by adapting on unlabeled test streams via self‑supervised objectives or entropy minimization.[15][16]
- Temporal consistency methods and TENT‑style approaches provide practical adaptation mechanisms for sequential data; evaluate stability given correlated inputs.[17][15]

### Uncertainty and calibration
- Use deep ensembles for spatiotemporal GNN/RNN forecasting to quantify epistemic uncertainty under domain shift and guide risk‑aware decisions.[18]
- Calibrate predictive distributions and multi‑horizon intervals with methods from traffic forecasting UQ surveys and ensemble calibration to maintain reliable SLAs across cities.[19][20][21][22]

### Data and splits
- Exploit LaDe’s multi‑city, multi‑month design to create leave‑one‑city‑out and season‑shift splits, measuring transfer and OOD robustness beyond in‑city i.i.d. splits.[3]
- Pair LaDe with external datasets (e.g., Amazon routes) using domain adaptation or shared encoders to test generalization from granular event logs to route‑level domains.[1][3]

### Practical recipe
- Start with a strong ERM baseline using a spatiotemporal encoder, then add domain adaptation (adversarial spatial + MMD temporal), pretrain with TS2Vec/TF‑C, and meta‑learn fast adapters; finally, add TTA and deep ensembles with calibration for deployment.[20][14][11][4][15][18][9]
- Report city‑held‑out results on LaDe’s splits, ablate each component, and tie improvements to invariance, adaptation, and calibration metrics for credible, reproducible generalization claims.[2][3]

[1](https://www.ijcai.org/proceedings/2022/0282.pdf)
[2](https://arxiv.org/pdf/2103.03097.pdf)
[3](https://arxiv.org/pdf/2306.10675.pdf)
[4](https://dl.acm.org/doi/10.1145/3534678.3539250)
[5](https://www.sciencedirect.com/science/article/abs/pii/S0950705125013917)
[6](https://arxiv.org/abs/1802.00386)
[7](https://openreview.net/pdf?id=Dc4rXq3HIA)
[8](https://proceedings.neurips.cc/paper_files/paper/2022/file/57568e093cbe0a222de0334b36e83cf5-Paper-Conference.pdf)
[9](https://github.com/mims-harvard/TFC-pretraining)
[10](https://openreview.net/pdf?id=OJ4mMfGKLN)
[11](https://arxiv.org/abs/2106.10466)
[12](https://arxiv.org/html/2502.06380v4)
[13](http://urban-computing.com/pdf/MetaLearning_tkde_2020.pdf)
[14](https://arxiv.org/pdf/1901.08518.pdf)
[15](https://epubs.siam.org/doi/pdf/10.1137/1.9781611978032.54?download=true)
[16](https://arxiv.org/pdf/2401.04148.pdf)
[17](https://openaccess.thecvf.com/content/CVPR2023W/ABAW/papers/Mutlu_TempT_Temporal_Consistency_for_Test-Time_Adaptation_CVPRW_2023_paper.pdf)
[18](https://arxiv.org/abs/2204.01618)
[19](https://www.sciencedirect.com/science/article/pii/S0010482523005619)
[20](https://arxiv.org/pdf/2208.05875.pdf)
[21](https://www.sciencedirect.com/science/article/abs/pii/S0952197624000290)
[22](https://vbn.aau.dk/files/747273383/306160.pdf)
[23](https://mediatum.ub.tum.de/doc/1779275/1779275.pdf)
[24](https://www.nature.com/articles/s41598-025-06586-6)
[25](http://proceedings.mlr.press/v139/mahajan21b/mahajan21b.pdf)
[26](https://www.sciencedirect.com/science/article/pii/S1110016825006477)
[27](https://kl4805.github.io/files/KDD22.pdf)
[28](https://openreview.net/forum?id=jeNWwtIX71)
[29](https://openreview.net/forum?id=y4F2YZxN9T)
[30](https://arxiv.org/html/2406.05628v1)
[31](https://arxiv.org/html/2507.07908)
[32](https://www.sciencedirect.com/science/article/pii/S2405959525001110)
[33](https://www.sciencedirect.com/science/article/abs/pii/S1746809423011114)
[34](https://www.sciencedirect.com/science/article/abs/pii/S0952197623019115)
[35](https://www.sciencedirect.com/science/article/pii/S0957417425018639)
[36](https://dl.acm.org/doi/10.1145/3627673.3680086)
[37](https://www.sciencedirect.com/science/article/pii/S0031320324006253)
