# SynRTP

## Reproducibility & Source Code

To ensure reproducibility and facilitate future research, we provide the complete source code of **SynRTP** in this anonymous repository.

* **SynRTP implementation.** The code for pre-training, GDRPO fine-tuning, and synergistic inference is available in `/algorithm/`.
* **Baseline implementations.** All baseline methods are included in `/baselines/`. The performance comparisons reported in our paper are fully reproducible under identical settings. Ready-to-run execution commands are listed in Section 6.5 of this repository.

---

## 1. Dataset Diversity & Generalization (Rebuttal Q1)

> Q1. Dataset diversity & generalization (R2-W1/W4/SI4, R3-W1/SI1, R4-W1).
>- Dataset. We add three datasets (strictly anonymized via hashed IDs and offset coordinates): Logistics-HZ/YT (distinct city topologies, Cainiao logistics) and Food-DL (food delivery, Ele.me platform). We also searched for international last-mile datasets, but public corpora currently lack the necessary task/trajectory fields. We will explicitly discuss this limitation and future plans.
>- Generalization. Across all five datasets, Logistics-SH/CQ/HZ/YT, Food-DL (renamed "TaskType–City"), SynRTP consistently outperforms the strongest RP/TP/RTP baselines: KRC improvements of 1.25–7.71% and LSD reductions of 8.00–17.55%, while MAE/RMSE decrease by up to 23.32%/22.39%. 

### 1.1 Extended Dataset

We have expanded the evaluation to `five` distinct datasets to verify robustness. The detailed statistics are provided in `Table 1`. The three newly added datasets are:

* **Cainiao Platform, Logistics:** `Logistics-HZ` (Hangzhou) and `Logistics-YT` (Yantai) from [`LaDe`](https://huggingface.co/datasets/Cainiao-AI/LaDe).
* **Ele.me Platform, Food Delivery:** [`Food-DL`](https://tianchi.aliyun.com/competition/entrance/231777/information) (Dalian).


<p align="left"> <b>Table&nbsp;1</b> Summary statistics of the datasets. AvgETA (in minutes) stands for the average arrival time per package. AvgPackage means the average package number of a courier per day. </p>

![Table 3](src/results_datasets.png)

#### A. Privacy Statement

All datasets are strictly anonymized. User IDs and order IDs are hashed, and GPS coordinates are offset to prevent re-identification while preserving topological properties.

#### B. Data Diversity

As shown in `Table 1`, these datasets cover a wide spectrum of business patterns, city scales, and urban environments.

**i) Business Logic Diversity.**
Across logistics and food-delivery scenarios, SynRTP is exposed to a wide range of routing and timing patterns.
* **Logistics:** `Logistics-SH/CQ/HZ/YT` follow batched, pre-planned multi-stop routes with high AvgPackage per courier and relatively flexible delivery windows (AvgETA>140 minutes), typical of parcel logistics. 
* **Food-Delivery:** In contrast, `Food-DL` represents high-frequency, point-to-point on-demand delivery with much smaller AvgPackage (4.0) and stricter time windows (AvgETA<30 minutes).


**ii) City Scale Diversity.**
To ensure the model generalizes across different administrative scales and population densities, we selected cities ranging from megacities to major regional hubs:

* **Mega-Cities (>20 Million):** Shanghai (SH) and Chongqing (CQ) represent the highest tier of urban density and complexity.
* **Large Metropolitan Area (10~20 Million):** Hangzhou (HZ) is a rapidly growing new-tier city with a population exceeding 10 million.
* **Major Regional Cities (5~10 Million):** Yantai (YT) and Dalian (DL) are important regional economic centers, testing the model’s adaptability to medium-to-large urban networks.

**iii) Urban Topology Diversity.**
The five cities also exhibit diverse urban forms and road-network topologies:

* **Shanghai (SH):** Flat megacity with a dense, grid-like road network and multiple commercial centers.
* **Chongqing (CQ):** Mountainous “multi-level” city with non-planar roads, steep gradients, and many bridges/tunnels.
* **Hangzhou (HZ):** Multi-center city combining e-commerce hubs with large scenic and preservation areas.
* **Yantai (YT) & Dalian (DL):** Coastal port cities with elongated, coastline-constrained urban belts.

These complementary topologies (grid-like vs. mountainous vs. coastal) jointly stress-test SynRTP under heterogeneous spatial constraints.

#### C. International Data Limitations

We conducted an exhaustive search for international last-mile datasets. However, existing public corpora (e.g., from Amazon or Grab) currently lack the **sequential trajectory fields** or **task-level timestamps** required for joint Route–Time Prediction (RTP). While we focus on high-quality industrial datasets from China, the structural diversity (e.g., logistics vs. food delivery, plain vs. mountain vs. coastal cities) provides a strong proxy for diverse global delivery scenarios.


### 1.2 Comprehensive Results Analysis

We present the full performance comparison across all five datasets in `Table 2`. 

<p align="left"> <b>Table&nbsp;2</b> Performance comparison on route and time prediction. An upward arrow ($\uparrow$) indicates that higher values are better, while a downward arrow ($\downarrow$) indicates that lower values are better. “--” denotes unavailable results. For SynRTP, we report the mean over three runs, with $\pm\mathrm{std}\le 0.003$ for KRC/HR@3, $\pm\mathrm{std}\le 0.02$ for LSD, and $\pm\mathrm{std}\le 0.26$ for MAE/RMSE. </p>

![Table 4](src/results_sh_cq.png)
![Table 5](src/results_food.png)


The results demonstrate strong and consistent generalization:
* **Consistent Improvements:** SynRTP consistently outperforms the strongest baselines (including *MRGRP* and *M2G4RTP*) across all datasets. Specifically, we observe **KRC improvements of 1.25%–7.71%**, **LSD reductions of 8.00%–17.55%**.
* **Cross-domain generalization (Food-DL).** The `Food-DL` dataset represents a fundamentally different business logic—point-to-point on-demand delivery with strict time windows, rather than planned multi-stop logistics routes.
    * *Result:* SynRTP achieves a **1.25% KRC gain**, **8.00% LSD reduction** and a massive **23.32% MAE reduction** on Food-DL.
    * *Insight:* This indicates that SynRTP's synergy mechanism does not overfit to logistics-specific patterns. The continuous time supervision effectively guides the policy on a highly dynamic food-delivery graph, supporting our hypothesis that **better time awareness leads to better routing**.


---

## 2. Computational Efficiency & Deployment (Rebuttal Q2)

>Q2. Computational efficiency and stability (R1-W2, R2-W5, R3-W3, R4-W2/W3).
>- Efficiency comparison. SynRTP uses only 0.2M parameters, 0.1GB GPU memory at inference, and achieves the fastest inference time (10.6–13.7s on the full test set) among RTP models, while MRGRP and M2G4RTP require up to 5.7M/2.2M parameters and significantly higher latency (15.6-254.4s).
>- Sampling-cost. Training-Time/Epoch scales approximately linearly with #sample and remains manageable at #sample=16, which we select as a good accuracy–efficiency tradeoff. RL fine-tuning is performed offline; online serving uses a single greedy decode, so latency is compatible with large web-scale logistics.
>- Stability. Repeated runs show negligible variance (std≤0.02 for KRC/HR@3/LSD), confirming SynRTP's stability.

<p align="center"> <b>Table&nbsp;2</b> Efficiency comparison across different datasets. All experiments use a fixed batch size of 64, and each model employs an identical configuration across datasets to ensure consistent parameter counts and GPU memory usage. Metrics reported include: number of parameters (Param., in millions), GPU memory consumption during training and inference (GPU.T. and GPU.I, in GB), training time per epoch (Train./Epoch, in seconds), and inference time on the full test set (Infer., in seconds). "--" indicates that the value is not available. Among the evaluated RTP methods, SynRTP achieves the best performance in terms of parameter scale, GPU memory usage during inference, and inference speed. </p>

![Table 2](src/results_eff.png)

### 2.1 Inference Efficiency & Deployment Costs

SynRTP is designed for **low-cost, high-frequency deployment**.

* **Model Lightweightness:** As shown in `Table 2`, SynRTP utilizes only **0.2M parameters** (Actor-only inference), which is **10$\times$~20$\times$ smaller** than graph-heavy baselines like *M2G4RTP* (2.2M) and *MRGRP* (5.7M).
* **Hardware Requirements:** During inference, SynRTP consumes only **0.1GB of GPU memory**. This extreme efficiency means SynRTP can be easily deployed on **consumer-grade GPUs (e.g., RTX 30/40 series)** or even **CPU-only inference servers** with minimal latency penalty, significantly lowering the deployment cost for logistics platforms.
* **Speed:** On the full test set, SynRTP achieves the fastest inference (10.6–13.7s total), fully satisfying real-time dispatching requirements.


### 2.2 Sampling Cost & Offline/Online Separation

We now clarify the impact of the sampling number (`#sample`) used in GDRPO.

* **Offline RL Training:** Multi-route sampling ('\#sample=16') is strictly an **offline training strategy**. During this phase, the computational cost scales linearly with `#sample`, as shown in **Figure 1**. We selected `#sample=16` as the optimal convergence-efficiency tradeoff.
* **Online Serving:** Once the model is deployed, the policy $\pi_\theta$ is frozen. We use **Greedy Decoding** (or single-path sampling) for inference. No group sampling or advantage calculation is performed online. Thus, the heavy RL computation does not impact the online system latency.


![Figure ](src/results_sample.png)

<p align="center"><b>Figure&nbsp;1</b> Computational overhead analysis on different datasets with varying sampling numbers $\#sample$. The training time per epoch (in seconds) exhibits a linear relationship with the increase of $\#sample$. While larger $\#sample$ values introduce more computation, the overhead remains within a manageable range across all datasets, justifying the choice of $\#sample=16$ for balancing efficiency and performance. </p>




## 3. Methodological Novelty: GDRPO vs. RL Baselines (Rebuttal Q3)

>Q3. Novelty of GDRPO vs. PPO/Sequence RL (R1-W1, R2-W3/SI1)
>- GDRPO is a task-specific adaptation of PPO/GRPO for routing that explicitly addresses the *granularity mismatch* between step-level policy updates and sequence-level evaluation metrics. We will add a formal comparison table (vs. PPO/GRPO/GSPO/ReST-RL) to highlight:
>- Critic-free design (vs. PPO). GDRPO removes the unstable and expensive value network (critic).
>- Stable Anchor (vs. GRPO/GSPO/ReST): Unlike Group Relative methods that use a noisy group mean as the baseline, GDRPO uses a deterministic greedy baseline ($\pi^*$) as a stable performance anchor. This design yields a low-variance Location-Deviation Advantage (Eq.17) that explicitly rewards beating the model's best deterministic strategy on the non-differentiable LSD metric.
>- Integration with multi-task learning. GDRPO is tightly coupled with the uncertainty-based multi-task loss (Eq. 21), which balances exploration with imitation.


To further clarify the novelty of **GDRPO**, we provide a multi-dimensional comparison with standard **PPO** and **GRPO**, as well as advanced sequence-level methods such as **GSPO** and **ReST-RL**.

**Table: Comparison of RL Optimization Algorithms for Routing**

| Feature Dimension | **PPO** | **GRPO** | **GSPO** [1] | **ReST-RL** [2] | **GDRPO (Ours)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Reward Granularity** | Token/Step-level | Token/Step-level | Sequence-level | Sequence-level | **Sequence-level (LSD)** |
| **Critic / Value Net** | **Required** (High Cost) | None | None | Optional / Value Model | **None** (Critic-free) |
| **Baseline Mechanism** | Learned $V_\phi(s)$ | Group Mean $\mathbb{E}[\pi^{(g)}]$ | Group Mean $\mathbb{E}[\pi^{(g)}]$ | Filtered History | **Greedy Anchor $\pi^*$** |
| **Advantage Estimation** | GAE (Step-wise) | Relative to Mean | Relative to Mean | Filtering / Scoring | **Location-Deviation** (Eq.17) |
| **Optimization Stability** | Low (Sensitive to Critic) | Moderate (Noisy Mean) | High | High | **Very High** (Deterministic) |
| **Best Use Case** | Continuous Control | LLM Reasoning | LLM Generation | Iterative Alignment | **Routing / Global Metrics** |

**Highlights of GDRPO:**
1.  **Granularity-Mismatch Solution.** Unlike PPO/GRPO, which perform step-level updates, GDRPO uses a **sequence-level advantage** to directly optimize the non-differentiable, holistic LSD metric.
2.  **Stable anchor baseline.** In contrast to GRPO/GSPO, which rely on a stochastic group-mean baseline, GDRPO uses a **deterministic greedy rollout ($\pi^*$)** as a stability anchor, reducing variance by explicitly asking: *“Did exploration beat the model’s current best deterministic strategy?”*
3.  **Computational efficiency.** Because GDRPO is critic-free, it avoids the substantial overhead of training a value network over complex graph-structured states.

---

## 4. Synergy Mechanism Theory (Rebuttal Q4)

>Q4. Synergy mechanism & gradient paths (R1-W3, R2-W4/SI5).
>- The synergy stems from the Route-Aware Context (Eq.14), which aggregates probability-weighted embeddings. The gradient path is: $\nabla \mathcal{L}_{Time} \to \text{Context} \to \text{Route Logits} (\pi_\theta)$. This path provides a dense, differentiable signal that shapes the policy to favor nodes enabling accurate time predictions, effectively using TP as an implicit reward.


We illustrate the gradient flow that allows the Time Prediction (TP) task to act as an implicit reward for the Route Prediction (RP) task.

* **Route-Aware Context:**
$$s_i = \sum_{j \in \mathcal{V}} \pi_\theta(j|\hat{\pi}_{<i}) \cdot \tilde{q}_j $$
where $s_i$ denotes the route-aware context at step $i$, and $\tilde{q}_j$ is the embedding of candidate node $j$.

* **Synergy Gradient Flow:**

$$\frac{\partial \mathcal{L}_{Time}}{\partial \theta} = \frac{\partial \mathcal{L}_{Time}}{\partial \hat{\delta}_i} \cdot \underbrace{\frac{\partial \hat{\delta}_i}{\partial s_i} \cdot \sum_{j} \tilde{q}_j}_{\text{Context Awareness}} \cdot \frac{\partial \pi_\theta(j|\dots)}{\partial \theta}$$
which makes explicit that gradients from the time loss propagate through the context $s_i$ and then back into the routing policy $\pi_\theta$.

* **Mechanism:** 
By backpropagating $\mathcal{L}_{\text{Time}}$ through the probability-weighted context $s_i$, the policy is encouraged to increase the probability of next-hop nodes that are spatially and temporally coherent. In effect, the time prediction loss acts as a dense, differentiable reward-shaping signal, complementing the sparse RL signal on the non-differentiable route metric.


---

## 5. Fairness, Related Work & Ethics (Rebuttal Q5)

>Q5. Fairness, related-work & ethics (R2-W2/W6/SI2; R3-W2; R4-W2).
>- Fairness. DutyTTE/MRGRP use official implementations with careful hyperparameter search; the other baselines follow the best configurations released with LaDe. All models are trained with the same input features, early stopping strategy, and evaluation metrics. Our Graphormer encoder uses only the same input features, without additional information. 
>- Related work. We will expand the discussion to cover additional RTP models.
>- Ethics. We will add a short ethics paragraph: SynRTP is currently used solely to predict couriers' future routes and arrival times to improve ETA reliability and dispatching, not to constrain individual workers' choices.

### 5.1 Expanded Related Work

We discuss the landscape of joint Route–Time Prediction (RTP) and position **SynRTP** relative to recent advances:

* **Advantages of joint RTP.** Joint modeling enables shared spatiotemporal representations, which can in principle benefit both RP and TP.
* **Existing Gaps:**
    * *RankETPA* and *DeepRoute+* treat the two tasks sequentially and are prone to error propagation.
    * *M2G4RTP* and *MRGRP* introduce strong graph encoders but still implement RP and TP as loosely coupled output heads, leading to an **“uncooperative problem”** (gradient isolation).
    * We additionally acknowledge **$I^2$RTP** (ICDE'23), which leverages community structures for delivery prediction. While $I^2$RTP enriches representations via community information, it continues to rely on conventional multi-task learning objectives. In contrast, **SynRTP** is, to our knowledge, the first to fundamentally alter the **optimization paradigm** via synergistic decoding and sequence-level RL, directly targeting the negative transfer issue.


### 5.2 Baseline Fairness

To ensure a fair comparison, all baselines (including *DutyTTE* and *MRGRP*) are re-trained under a unified protocol:

1. **Identical features.** All models consume the same raw inputs (e.g., coordinates and timestamps). SynRTP feeds these features into our Graphormer encoder, while baselines retain their original encoder architectures unless they require specialized inputs.  
2. **Identical evaluation.** We use the same train/validation/test splits and metric calculation procedures for all methods.  
3. **Open-source Implementations.** As mentioned in the start, all re-implementations are released to guarantee transparency and full reproducibility.

### 5.3 Ethical Considerations

* **Assistance vs. control.** SynRTP is designed to **predict** natural courier behavior to improve system estimates (ETA), not to **prescribe** or enforce specific routes. It serves as an auxiliary tool to reduce dispatcher uncertainty.  
* **Cognitive load and well-being.** By predicting routes that better align with human preferences (e.g., avoiding difficult U-turns or overly complex detours), the system can assign tasks in a way that helps reduce physical and cognitive fatigue for couriers.  
* **Privacy protection.** All courier IDs are hashed, and GPS coordinates are transformed relative to the region center, ensuring that no individual can be directly tracked or re-identified from the released data.


---















## 6. Experimental Details
---
### 6.1 Implementation Details & Fairness Protocol

To ensure reproducibility and a rigorous fair comparison, all experiments are conducted on a unified hardware platform with a single Tesla V100 GPU (16 GB). SynRTP is implemented in PyTorch. For all baseline models, we adopt a standardized evaluation protocol to avoid implementation bias:

**(1) Standardized benchmark configurations**

* **[`LaDe`](https://huggingface.co/datasets/Cainiao-AI/LaDe) benchmark baselines.** Most baselines (including DeepRoute, Graph2Route, etc.) and the datasets used in this paper are taken from the open-source LaDe benchmark repository. To make our results directly comparable with community standards, we strictly use the official implementations and their default optimal hyperparameter settings provided in LaDe.  
* **Independent baselines.** For baselines not included in LaDe (e.g., DutyTTE and MRGRP), we use their official open-source implementations and adopt the default optimal hyperparameter combinations recommended by the original authors. This strategy ensures that every baseline is evaluated close to its intended peak performance, avoiding bias from subjective re-tuning.

**(2) Strict fairness control**

Beyond model configurations, we enforce a unified training protocol across all methods so that no model receives an unfair advantage.  
- **Input consistency.** All models use exactly the same set of input features (spatial coordinates, temporal timestamps, and courier profiles). No baseline is handicapped by missing features, and no model has access to additional information unavailable to others.  
- **Termination criterion.** To prevent over-training or under-training biases, we apply a consistent early-stopping mechanism to all models: training stops if the validation metric (KRC) does not improve for 11 consecutive epochs.

**(3) SynRTP settings**

For SynRTP, hyperparameters are selected based on validation performance: the hidden dimension is set to $d_h = 32$, the Graphormer encoder has 3 layers with 4 attention heads, and the GDRPO group sampling size is $G = 16$. We train the model in a two-stage scheme using the Adam optimizer with a learning rate of $1 \times 10^{-4}$.



### 6.2 Dataset Description

We evaluate our approach using five large-scale real-world datasets to ensure robust generalization. Beyond the original **logistics datasets** from `Shanghai` and `Chongqing` (collected by Cainiao, [link](https://huggingface.co/datasets/Cainiao-AI/LaDe)), we incorporate two additional logistics datasets from `Hangzhou` and `Yantai`, as well as a cross-domain **food delivery dataset** from `Dalian` (collected by Ele.me, [link](https://tianchi.aliyun.com/competition/entrance/231777/information)). Collectively, these datasets span diverse urban environments (from mountainous terrains to coastal cities) and distinct operational modes (standard logistics vs. on-demand food delivery), providing a comprehensive benchmark for performance evaluation.





### 6.3 Data Generation for Model Training

Install environment dependencies using the following command:

```shell
pip install -r requirements.txt
```

After downloading the original datasets, please use the following command to generate the data required for model training:
```shell
bash DataPipeline.sh
```

To facilitate verification of the correctness of the model code, we provide a very small dataset of Logistics-YT, extracting a batch size of 8 from each of the original data training set, validation set and test set (the default batch size of the model dataset is 64).


### 6.4 Training SynRTP Model


Taking the Logistics-YT dataset as an example. Run the following command to train the SynRTP. 

```shell
python run.py --dataset yt_dataset
```




### 6.5 Baseline Reproduction

Taking the Logistics-YT dataset as an example. Use the following commands to reproduce baseline models:
```shell
# Time-Greedy
python baselines/LaDe/route_prediction/run.py --model Time-Greedy --dataset yt_dataset

# Distance-Greedy
python baselines/LaDe/route_prediction/run.py --model Distance-Greedy --dataset yt_dataset

# Osquare
python baselines/LaDe/route_prediction/run.py --model Osquare --dataset yt_dataset

# DeepRoute
python baselines/LaDe/route_prediction/run.py --model DeepRoute --dataset yt_dataset

# Graph2Route
python baselines/LaDe/route_prediction/run.py --model Graph2Route --dataset yt_dataset

# DRL4Route
python baselines/LaDe/route_prediction/run.py --model DRL4Route --dataset yt_dataset

# Static-ETA
python baselines/LaDe/time_prediction/run.py --model Static-ETA --dataset yt_dataset

# KNN-MultiETA
python baselines/LaDe/time_prediction/run.py --model KNN-MultiETA --dataset yt_dataset

# XGB-MultiETA
python baselines/LaDe/time_prediction/run.py --model XGB-MultiETA --dataset yt_dataset

# DeepETA
python baselines/LaDe/time_prediction/run.py --model DeepETA --dataset yt_dataset

# DutyTTE
python baselines/DutyTTE/main.py --dataset_name yt_dataset

# RankETPA
python baselines/LaDe/time_prediction/run.py --model RankETPA --dataset yt_dataset

# M2G4RTP
python baselines/LaDe/route_prediction/run.py --model M2G4RTP --dataset yt_dataset

# MRGRP
python baselines/MRGRP/run.py --dataset_name yt_dataset

```


