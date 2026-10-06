# Anomaly Heterogeneity Learning for Open-set Supervised Anomaly Detection
**CVPR 2024**  
Authors: Jiawen ZHU, Choubo DING, Yu TIAN, Guansong PANG  
Institution: Singapore Management University  
Paper Link: [https://ink.library.smu.edu.sg/sis_research/9760](https://ink.library.smu.edu.sg/sis_research/9760)  
arXiv:2310.12790v3

## Abstract
Open-set supervised anomaly detection (OSAD) – a recently emerging anomaly detection area – aims at utilizing a few samples of anomaly classes seen during training to detect unseen anomalies (i.e., samples from open-set anomaly classes), while effectively identifying the seen anomalies. Benefiting from the prior knowledge illustrated by the seen anomalies, current OSAD methods can often largely reduce false positive errors. However, these methods are trained in a closed-set setting and treat the anomaly examples as from a homogeneous distribution, rendering them less effective in generalizing to unseen anomalies that can be drawn from any distribution.

This paper proposes to learn heterogeneous anomaly distributions using the limited anomaly examples to address this issue. To this end, we introduce a novel approach, namely Anomaly Heterogeneity Learning (AHL), that simulates a diverse set of heterogeneous anomaly distributions and then utilizes them to learn a unified heterogeneous abnormality model in surrogate open-set environments. Further, AHL is a generic framework that existing OSAD models can plug and play for enhancing their abnormality modeling.

Extensive experiments on nine real-world anomaly detection datasets show that:

1. AHL can substantially enhance different state-of-the-art OSAD models in detecting seen and unseen anomalies;
2. AHL has strong generalization ability to unseen anomalies in new domains.

Code repository: [https://github.com/mala-lab/AHL](https://github.com/mala-lab/AHL)

---

## 1. Introduction
### 1.1 Background of Anomaly Detection
Anomaly detection (AD) identifies samples significantly deviating from majority normal data, widely applied in industrial inspection, medical imaging, etc.

Most classic AD methods are **one-class unsupervised schemes**: only normal samples for training, no defect labels. Their obvious drawback: lack anomaly prior knowledge, leading to high false positive rates.

### 1.2 Open-set Supervised Anomaly Detection (OSAD) Definition
In many real scenarios, a small number of historical defective samples are available. OSAD targets this setting:

+ Train with limited seen anomaly samples;
+ Not only detect defects similar to training anomalies, but also generalize to **unseen novel anomaly types** (open-set property).

Existing OSAD core flaw: All training anomalies are regarded as a single homogeneous distribution. In reality, defects are highly diverse in shape, texture, position, leading to poor generalization to unseen anomaly categories.

### 1.3 Core Idea of AHL
1. Generate multiple heterogeneous anomaly sub-datasets by clustering normal samples and mixing with sampled real/pseudo anomalies;
2. Train multiple base models on different sub-datasets to capture diverse abnormal patterns;
3. Collaboratively aggregate knowledge from all base models to train one unified global detector under simulated open-set validation split;
4. Self-supervised weight estimator dynamically adjust the contribution of each base model according to generalization performance.

### 1.4 Four Main Contributions
1. **Novel Framework AHL**: First framework to explicitly model heterogeneous anomaly distributions for OSAD, solving the homogeneous distribution assumption limitation;
2. **Instantiable Model**: Propose collaborative differentiable learning (CDL) to iteratively train base models and unified global model with surrogate open-set validation;
3. **Plug-and-play Genericity**: AHL can be mounted on existing OSAD backbones (DevNet, DRA) to consistently boost performance;
4. **Strong Generalization**: SOTA results on 9 industrial/medical datasets, including cross-domain unseen anomaly scenarios.

![]()  
_Figure 1: Current OSAD vs Our AHL. (a) Traditional methods treat all anomalies as one single homogeneous cluster. (b) AHL simulates multiple heterogeneous anomaly distributions to learn diverse abnormal patterns._

+ Circle: Normal sample
+ Triangle: Seen anomaly sample
+ Square: Pseudo anomaly sample

---

## 2. Related Work
### 2.1 Unsupervised Anomaly Detection
Only normal samples for training:

+ One-class classification (OC-SVM, SVDD): learn compact normal boundary;
+ Reconstruction-based (AE, GAN, DRAEM): anomalies have higher reconstruction loss;
+ Knowledge distillation / Self-supervised methods (CutPaste, PatchCore).

Limitation: No anomaly supervision, high false alarm rate in complex scenes.

### 2.2 Open-Set Supervised Anomaly Detection (OSAD)
Use small labeled defect sets:

1. DevNet: Propose one-sided Deviation Loss to separate normal/abnormal score distribution;
2. DRA: Disentangle seen, latent, residual anomaly representations;
3. BGAD, PRN, UBnormal, OpenVAD: Extend OSAD to image/video.

Common defect of above methods: Assume all training anomalies belong to the same single distribution, weak generalization to totally new defect types.

### 2.3 Ensemble Comparison Note
Simple ensemble stacking cannot replace AHL collaborative learning: independent ensembles ignore shared & distinct patterns across anomaly distributions, resulting in suboptimal generalization.

---

## 3. Anomaly Heterogeneity Learning (AHL)
### 3.1 Problem Statement
Input image dataset with binary labels $ y_i \in \{0(\text{normal}),1(\text{anomaly})\} $

+ $ X_n $: Feature set of normal images ($ N \gg M $);
+ $ X_a $: Feature set of seen training anomalies;
+ Task: Train scoring function $ g(X)\in\mathbb{R} $, assign higher scores to all anomalies (seen + unseen).

OSAD open-set constraint: Training anomaly classes $ S $ are only a subset of all possible test anomaly classes $ C, S\subset C $.

AHL pipeline two core modules:

1. HADG: Heterogeneous Anomaly Distribution Generation
2. CDL: Collaborative Differentiable Learning

![]()  
_Figure 2: Overall AHL Pipeline_

1. HADG generate T heterogeneous sub-datasets $ \{D_i\} $;
2. Each $ D_i $ split into support set $ D_i^s $ (train base model) + query set $ D_i^q $ (simulate unseen open-set data);
3. Train T base models $ \phi_i $ on each $ D_i^s $;
4. Use query-set losses to collaboratively update unified global model $ g $;
5. Self-supervised module $ \psi $ compute dynamic importance weight $ w_i $ for each base model;
6. Iteratively circulate base model training & global model optimization.

### 3.2 Heterogeneous Anomaly Distribution Generation (HADG)
Goal: Generate T sub-datasets with distinct normal & anomaly composition, to simulate diverse abnormal distributions.  
Step1: Cluster all normal features into $ C $ clusters via k-means (default $ \(C=3\) $).  
Step2 For each sub-dataset $ D_i $:

1. Randomly select one normal cluster as $ X_{n,i} $ (unique normal mode per subset);
2. Random sample real seen anomalies + generated pseudo anomalies (CutMix / CutPaste / DRAEM Mask) as $ X_{a,i} $;
3. Split $ D_i = X_{n,i}\cup X_{a,i} $ into disjoint $ D_i^s $ (support train) & $ D_i^q $ (query unseen validation):
    - Normal clusters of $ D_i^s $ and $ D_i^q $ are different;
    - Anomaly samples in support & query set have zero overlap, to build real open-set simulation.

Hyperparameter: Default total subsets $ \(T=7\) $.

### 3.3 Collaborative Differentiable Learning (CDL)
CDL two-stage alternating optimization: train T base models, then aggregate all base knowledge to update unified global model $ g $.

#### Step 1: Train T Heterogeneous Base Models $ \phi_i $
Each base model $ \phi_i $ shares the same backbone as the base OSAD model (DevNet/DRA), optimized with Deviation Loss on its own support set $ D_i^s $:

$ \mathcal{L}_{\phi_{i}}=\sum_{x_j\in D_i^s}\ell_{dev}(\phi_i(x_j),y_j) $

$ \ell_{dev} $: Deviation Loss from DevNet, push normal score near 0, anomaly score over margin $ m $.

Each $ \phi_i $ only learns one partial abnormal distribution, so single base model cannot handle all defect types.

#### Step 2: Collaborative Global Model Optimization
All base models fixed; use open-set query set losses to update unified model $ g $:

$ \mathcal{L}_{cdl}=\sum_{i=1}^T\sum_{x_j\in D_i^q}\mathcal{L}_{\phi_i}(\phi_i(x_j),y_j) $

After one global update epoch, copy all weights of $ g $ to every base model $ \phi_i $ for next round iteration:

$ \theta_i^{t+1} \leftarrow \theta_g^t $

This realizes bidirectional knowledge circulation: base models learn local heterogeneous defects, global model integrates universal abnormal logic, then feed back to all base models.

#### Dynamic Base Model Importance Estimation $ \psi $
Different $ D_i $ have varying quality; bad base models should have low weight during aggregation.

1. Sequential network (BiLSTM) $ \psi $ take historical score sequence of all base models to predict next epoch scores (self-supervised MSE loss $ \mathcal{L}_{seq} $);
2. Compute generalization error $ r_i^t $ for each $ \phi_i $ via prediction deviation vs ground truth;
3. Normalize inverse error as importance weight $ w_i^t $:

$ w_{i}^{t}=\frac{\exp(-r_{i}^{t})}{\sum_{i=1}^{T}\exp(-r_{i}^{t})} $

4. Weighted aggregated CDL loss:

$ \mathcal{L}_{cdl}^{+}=\sum_{i=1}^T w_i^t \sum_{x_j\in D_i^q}\mathcal{L}_{\phi_i}(\phi_i(x_j),y_j) $

### Algorithm 1 Full AHL Training Pipeline
```plain
Input: Full dataset D={x,y}, base model template φ, sequential estimator \(\psi\)
1. // Stage1 HADG Generate T heterogeneous sub-datasets {D_i}
2. // Iterative Collaborative Training
for epoch = 1 to MaxEpoch do
    # Substep1: Update all T base models with support set D_i^s
    for i=1 to T:
        Update θ_i by L_phi on D_i^s
    # Substep2: Compute dynamic importance weight w_i via \(\psi\)
    if epoch >=5:
        Compute generalization error r_i for each φ_i, calculate w_i
    else:
        Equal weight w_i = 1/T
    # Substep3: Update unified global model g with weighted query loss
    Update θ_g by L_cdl^+ on all D_i^q
    # Knowledge backflow: sync g's weight to all base models
    All θ_i ← θ_g
end for
Output: Unified anomaly detector g (discard all T base models during inference)
```

---

## 4. Experiments
### 4.1 Experimental Setup
#### Datasets
9 real-world AD datasets covering industrial texture/object, medical, aerospace:

1. Industrial: MVTec AD, AITEX, SDD, ELPV, Optical
2. Aerospace: Mastcam
3. Medical: HeadCT, BrainMRI, Hyper-Kvasir

Two evaluation protocols:

1. General Setting: Random sample multiple anomaly classes as training seen defects;
2. Hard Setting: Train with only one single anomaly class, test on all other unseen defect types (extreme open-set).

Two few-shot protocols: M=10 seen anomalies / M=1 single seen anomaly.

#### Baselines
SAOE, MLEP, FLOS, DevNet, DRA  
AHL instantiation: AHL(DevNet), AHL(DRA) (plug AHL on DevNet/DRA backbone).

#### Hyperparameters
+ Normal cluster count $ \(C=3\) $; Subset number $ \(T=7\) $;
+ BiLSTM hidden dim=7, sequential input length $ \(K=5\) $;
+ Optimizer Adam: base model lr=2e-4, global model lr=5e-3, (\psi) lr=0.02;
+ Pseudo anomaly augment pool: CutMix / CutPaste / DRAEM Mask random pick per subset.

### 4.2 Main Results (General Setting)
Table1 AUC mean±std (10-shot / 1-shot)  
All datasets AHL(DevNet) / AHL(DRA) consistently outperform vanilla DevNet/DRA, average AUC gain up to 9% for DevNet, 4% for DRA.

### 4.3 Hard Setting (Cross-Anomaly Generalization)
Table2: Train on single anomaly class, test on totally unseen defect categories.  
AHL still brings stable AUC lift, proving strong open-set generalization to brand-new anomaly patterns never seen in training.

### 4.4 Cross-Domain Generalization Test
Train on source MVTec texture category, test on unseen target texture domains.  
AHL(DRA) largely outperforms raw DRA, cross-domain AUC nearly matches same-domain performance.

### 4.5 Ablation Studies
1. HADG Necessity: Random subset ensemble (RamHADG) performance lower than official HADG heterogeneous partition;
2. CDL Necessity: Remove collaborative query loss → obvious AUC drop;
3. Dynamic Weight (\psi): Equal weight simplified CDL⁻ degrade results on hard datasets (AITEX, Mastcam);
4. Hyperparameter Sensitivity:
    - Too many normal clusters C cause small cluster overfitting; optimal (C=3);
    - Sequence length (K=5) achieves best balance of cost & performance.

![]()  
_Figure4 DevNet vs AHL(DRA) with/without pseudo anomaly augmentation. AHL still gains clear improvement without augmentation._

### 4.6 Comparison with Unsupervised AD
AHL(DRA) beats PatchCore/KDAD fully unsupervised methods by large margin, while close to fully supervised FS-DRA (all anomaly types provided for training).

---

## 5. Conclusion
This paper proposes Anomaly Heterogeneity Learning (AHL), a generic OSAD framework targeting the homogeneous anomaly distribution flaw of prior methods.

1. HAD module generates diverse heterogeneous anomaly sub-datasets to simulate abundant abnormal patterns;
2. CDL alternately trains multiple base detectors and a unified global model, with self-supervised dynamic weight balancing;
3. Extensive industrial/medical experiments validate AHL significantly improves seen & unseen anomaly detection performance, with strong cross-domain transfer ability.

Future work: Optimize pseudo anomaly generation strategy, extend AHL to pixel-level segmentation tasks.

---

## References
[1] Andra Acsintoae et al. Ubnormal, CVPR2022  
[2] Samet Akcay et al. Ganomaly, ACCV2018  
[3] Marcella Astrid et al. Pseudobound, Neurocomputing 2023  
[4] Liron Bergman et al. Classification-based anomaly detection, arXiv2020  
[5] Paul Bergmann et al. MVTec AD, CVPR2019  
[6] Tri Cao et al. Anomaly detection under distribution shift, ICCV2023  
[7] Hanna Borgli et al. HyperKvasir, Scientific Data 2020  
[8] Jakob Bozic et al. Mixed supervision for surface-defect detection, Comput. Ind. 2021  
[9] Yuanhong Chen et al. Deep one-class classification, AAAI2022  
[10] Wen-Hsuan Chu et al. Neural batch sampling, ECCV2020  
[11] Yingxian Chen et al. MGFN, AAAI2023  
[12] Choubo Ding et al. DRA, CVPR2022  
[13] Sergiu Deitsch et al. ELPV dataset  
[14] Hanqiu Deng et al. Reverse distillation, CVPR2022  
[15] Sangdoo Yun et al. CutMix, ICCV2019  
[16] Chun-Li Li et al. CutPaste, CVPR2021  
[17] Vitjan Zavrtanik et al. DRAEM, ICCV2021  
[18] Guansong Pang et al. DevNet, arXiv2021  
[19] Guansong Pang et al. Survey Deep Anomaly Detection, ACM CSUR 2021  
[20] Karsten Roth et al. PatchCore, CVPR2022  
[21] Lukas Ruff et al. Deep semi-supervised AD, ICLR2020  
[22] All other cited papers omitted for brevity

---

# Appendix (Abbreviated)
## A Dataset Statistics
Table6: Full normal/anomaly sample count for MVTec, AITEX, SDD, ELPV, Mastcam, BrainMRI, HeadCT, Hyper-Kvasir.

## B Implementation Details
1. HADG sampling rule for support/query split;
2. Three pseudo anomaly generation pipelines;
3. Complete mathematical expression of Deviation Loss.

## C Full Experiment Tables
Table7 General setting full category AUC  
Table8 Hard setting single-anomaly train AUC  
Table9 Class-level ablation results  
Table10 Overlap ablation on support/query split

