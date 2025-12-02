# SynRTP

The architecture of SynRTP, as shown in Figure 1, comprises three key components: 1) A **spatiotemporal graph encoder** that captures both spatial dependencies among tasks and their temporal evolution. 2) A **synergistic route-time decoder** where the route policy and time predictor are jointly optimized through gradient cooperation (addressing gradient isolation). 3) A **RL-enhanced training strategy** combining GDRPO for enhanced route exploration with uncertainty-based multi-task balancing. test

![Figure 1](./src/model.png)

<p align="center"><b>Figure&nbsp;1</b> Architecture of SynRTP.</p>


## 1. Supplementary Experiment

---

### 1.1 Model Generalization on New Dataset

To demonstrate the robust generalization capability of our proposed framework beyond the original experimental settings, we extended our evaluation to include three additional large-scale real-world datasets. These datasets were strategically selected to introduce significant diversity in terms of service platforms, business scenarios, and geographic environments:

Food-DL (Cross-Platform & Cross-Scenario): Sourced from Ele.me, a leading on-demand food delivery platform. Unlike the original logistics datasets (LaDe/Cainiao), this dataset represents a distinct Food Delivery scenario with different time sensitivities and routing constraints, verifying the model's effectiveness across different platforms and business modes.

Logistics-HZ & Logistics-YT (Geographic Generalization): Sourced from two additional major cities (Hangzhou and Yantai). These datasets validate the model's adaptability to varying urban layouts and road networks.

As shown in Table R1, SynRTP consistently outperforms state-of-the-art baselines across all three new datasets. Notably, in the cross-domain Food-DL scenario, our model achieves a substantial performance gain, reducing the MAE of time prediction by 23.32% compared to the best baseline while maintaining superior route prediction accuracy. These results confirm that SynRTP generalizes well to diverse delivery environments and is not limited to specific platforms or city patterns.

<p align="center">   <b>Table 1</b> Performance comparisons on three additional datasets (Food-DL, Logistics-HZ, and Logistics-YT). </p>

![Table 1](src/results_food.png)


### 1.2 Computational Efficiency and Scalability Analysis

To address concerns regarding the computational complexity and deployment feasibility of our framework, we conducted a comprehensive efficiency analysis across all five datasets. We focus on two key aspects: inference efficiency (crucial for real-time online deployment) and training scalability (specifically regarding the sampling size $G$).

<b>(1) Inference Efficiency and Resource Usage</b>

As shown in Table R2, we compared SynRTP with representative baselines. Lightweight Architecture: SynRTP exhibits remarkable model compactness with only 0.2M parameters, significantly fewer than other joint RTP models (e.g., MRGRP requires 5.7M). Fast Inference Speed: Crucially for real-time logistics systems, our model achieves SOTA-level inference efficiency (e.g., 10.6s for the entire test set on Food-DL), which is comparable to simple greedy heuristics and significantly faster than complex deep learning baselines like Graph2Route or MRGRP. Low Memory Footprint: During the inference phase, the GPU memory consumption is minimal (0.10 GB), making the model highly suitable for deployment on resource-constrained edge devices or high-concurrency cloud environments. It is important to note that while the reinforcement learning process (GDRPO) increases training time, this is strictly an offline cost that does not impact online service latency.

<p align="center"> <b>Table 2</b> Efficiency comparison across different datasets. </p>

![Table 2](src/results_eff.png)


<b> (2) Scalability of Sampling Size $G$ </b>

We further investigated the computational overhead introduced by the group sampling mechanism during the GDRPO phase. Figure R1 illustrates the relationship between training time per epoch and the sampling size $G$. The results demonstrate a Linear Scalability, where training time increases linearly rather than exponentially with $G$, indicating that the computational cost is predictable and controllable. Based on this analysis, we selected $G=16$ as the default setting to achieve an optimal balance between exploration sufficiency and training efficiency. Even with this setting, the training overhead remains within a manageable range, while the inference stage remains completely unaffected by the value of $G$.

![Figure 1](src/results_sample.png)

<p align="center"><b>Figure 1</b> Computational overhead analysis with varying sampling numbers <i>G</i> </p>



### 1.3 Theoretical Analysis: The Relationship between GDRPO, PPO, and GRPO






### 1.4 Statistics on Datasets

To ensure a comprehensive evaluation, we expanded our experimental scope to include five large-scale real-world datasets, which cover a wide spectrum of delivery scenarios, ranging from standard package logistics to high-urgency food delivery.

<b>(1) Logistics Datasets (Cross-City Diversity)</b>

The Logistics-SH, Logistics-CQ, Logistics-HZ, and Logistics-YT datasets are collected from the Cainiao Network. They span a substantial period of 6 months and cover four distinct Chinese cities: Shanghai (mega-metropolis), Chongqing (mountainous terrain), Hangzhou (dense urban), and Yantai (coastal city). This geographic diversity challenges the model with varying road network topologies and traffic densities.

<b>(2) Food Delivery Dataset (Cross-Domain Diversity)</b>

The Food-DL dataset, sourced from Ele.me, introduces a fundamentally different operational mode. As shown in the table, this dataset exhibits a significantly shorter AvgETA (27 minutes vs. 150 minutes for logistics) and a smaller AvgPackage (4.0 vs. 15.0). These statistics reflect the "instant" nature of food delivery, which requires the model to handle high-frequency, time-sensitive tasks with strict constraints, differing sharply from the batched delivery patterns in logistics.

The inclusion of these diverse datasets confirms that our proposed SynRTP framework is not overfitted to a specific platform or city but is robust across varying business logic and operational scales.The dataset statistics are summarized in **Table 3**.

<p align="center"> <b>Table 3</b> Summary statistics of the datasets. AvgETA (in minutes) stands for the average arrival time per package. AvgPackage means the average package number of a courier per day. </p>

![Table 3](src/results_datasets.png)






## 2. Experimental Details

### 2.1 Experimental Setting


SynRTP is implemented in PyTorch and trained on a Tesla V100 (16 GB) GPU. After extensive hyperparameter tuning(details are provided in Sec 4.4 of the paper.), we select a hidden dimension $d_h=32$, a 3‑layer Graphormer encoder with 4 attention heads per layer, and a GDRPO group sampling size $G=16$. Policy updates use the Adam optimizer with a learning rate of $1\times10^{-4}$ and a PPO‑style clipping parameter $\epsilon=0.2$. Training adopts a two‑stage scheme: 4 epochs of supervised pre‑training followed by fine‑tuning with a batch size of 64. A cosine annealing scheduler controls the learning rate, and early stopping is applied with a patience of 11 epochs based on the validation KRC metric.



### 1.2 Dataset Description

We evaluate our approach using two large-scale real-world last-mile delivery datasets from LaDe[^1], collected by Cainiao Network, one of China's largest logistics platforms. The datasets comprise delivery records from ***Shanghai*** and ***Chongqing***, representing diverse urban environments. The **Table 1** summarizes key statistics. Each dataset spans six months and covers approximately 400 km<sup>2</sup>, with couriers serving as workers and delivery tasks as nodes in our formulation.

[^1]: https://huggingface.co/datasets/Cainiao-AI/LaDe

<p align="center"><b>Table&nbsp;1</b> Dataset statistics.</p>
Statistics of the two subsets from LaDe used in our experiments. AvgETA stands for the average arrival time per package. AvgPackage means the average package number of a courier per day. The unit of AvgETA is minutes.  

![Table 1](src/dataset.png)



The original datasets can be downloaded from the following link: https://huggingface.co/datasets/Cainiao-AI/LaDe. 

Install environment dependencies using the following command:

```shell
pip install -r requirements.txt
```

After downloading the original datasets, please use the following command to generate the data required for model training:
```shell
python data_processing.py
```

To facilitate verification of the correctness of the model code, we provide a very small dataset, extracting a batch size of 8 from each of the original data training set, validation set and test set (the default batch size of the model dataset is 64). The data structure should be like:
/data/dataset/

├── cq_dataset    
│   ├── train_small.npy   
│   └── ...    
└── sh_dataset  
    ├── train_small.npy  
    └── ...  



### 1.3 Training SynRTP Model


Run the following command to train the SynRTP:

```shell
python run.py
```


### 1.4 Experimental Results

<p align="center">
<b>Table&nbsp;2</b> Performance comparison on route and time prediction. The upward arrow (↑) indicates that a higher value is better for metrics; the downward arrow (↓) indicates that a lower value is better for metrics. '--' means not available.
</p>

<p align="center"><b>Table&nbsp;2</b> Performance comparisons.</p>

![Table 2](src/results.png)



### 1.5 Baseline Reproduction

The baseline reproduction method can be obtained through the following link：
https://github.com/wenhaomin/LaDe/tree/master


