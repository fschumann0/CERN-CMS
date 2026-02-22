# A Deep Learning Approach to Classifying Z Boson Candidates in Proton-Proton Collisions Using Dielectron Events

<img width="1404" height="844" alt="image" src="https://github.com/user-attachments/assets/42bff86e-ffec-4e1a-9506-809bf4816fe8" />


## Introduction

This repository contains the data analysis and deep learning pipeline I built to classify dielectron collision events and identify potential Z boson candidates.

The goal of this project was to develop an end-to-end machine learning workflow using real data from the CMS Collaboration at CERN, and train a neural network capable of distinguishing events consistent with Z → e⁺e⁻ decays from background events.

This was one of my first complete deep learning implementations applied to real high-energy physics data, and I’m very happy with how it turned out, especially achieving  almost 99% classification accuracy after fine-tuning.

## Data

This data was sourced from the CERN Open Data Portal: Dielectron events recorded by CMS (2010): https://opendata.cern.ch/record/304

Specifically, this dataset contains 100.000 reconstructed events where two electrons were detected in proton–proton collisions. Each event contains kinematic properties of two electrons and the invariant mass of the pair. Note: These data were selected for use in education and outreach and contain a subset of the total event information. The selection criteria may be different from that used in CMS physics results.

**Features used**

For each event:

1. Run: Run number

2. Event: Event ID

3. E1, px1, py1, pz1: Four-momentum components of electron 1

4. pt1: Transverse momentum of electron 1

5. eta1: Pseudorapidity of electron 1

6. phi1: Azimuthal angle of electron 1

7. Q1: Charge of electron 1

8. E2, px2, py2, pz2: Four-momentum components of electron 2

9. pt2: Transverse momentum of electron 2

10. eta2: Pseudorapidity of electron 2

11. phi2: Azimuthal angle of electron 2

12. Q2: Charge of electron 2

13. M: Invariant mass of the electron pair

## Classification Target

The task is binary classification:

Label = 1: Event is a Z boson candidate

Label = 0: Background event

We define a Z boson candidate using the invariant mass window 80 GeV ≤ M ≤ 100 GeV. The nominal Z boson mass is approximately M_Z ≈ 91 GeV. Events inside this mass window are consistent with Z → e⁺e⁻ decay.

## Feature Engineering

The dataset was already clean. No preprocessing was required beyond scaling.

However, I engineered additional physics-motivated features:

1. delta_eta = eta1 - eta2

2. delta_phi = phi1 - phi2

3. delta_R = sqrt(delta_eta² + delta_phi²)

4. pt_ratio = pt1 / pt2

5. pt_sum = pt1 + pt2

6. pt_diff = |pt1 - pt2|

7. cos_theta: cosine of the 3D angle between the two electron momenta

These features help the model capture geometric and kinematic correlations between the electrons.

## Model Architecture

The final model was built using TensorFlow / Keras (Sequential API).

**Architecture:**

Dense (256) + BatchNorm + Dropout

Dense (128) + BatchNorm + Dropout

Dense (64) + Dropout

Output layer (Sigmoid)

**Regularization:**

L2 weight decay

Dropout

Early stopping

**Optimizer:**

Adam

**Loss function:**

Binary crossentropy

**Class imbalance was handled using:**

class_weight='balanced'

Threshold tuning was applied to maximize classification accuracy.

## Results

**After fine-tuning:**

Accuracy: ~98–99%

High F1-score

Strong ROC AUC

Clear separation around M ≈ 91 GeV

The model learns a sharp probability peak centered on the Z boson mass region.

**Benchmark comparisons were performed against:**

1. Majority class baseline

2. Logistic regression

The neural network significantly outperformed both.

# Benchmarks

Benchmarks were included to validate that the neural network truly learns meaningful physics patterns.

**Models compared:**

1. Majority class predictor

2. Logistic Regression

3. Fine-tuned Neural Network

**Metrics used:**

Accuracy

F1-score

ROC AUC

Confusion matrices

Probability vs invariant mass plots

# References 

- Mahesh, Batta. (2019). Machine Learning Algorithms -A Review. 10.21275/ART20203995.
- Deep learning with python. (2017). François Chollet. Manning Publications.

ML meets Particle Physics:

- de Oliveira, L., Kagan, M., Mackey, L. et al. Jet-images — deep learning edition. J. High Energ. Phys. 2016, 69 (2016). https://doi.org/10.1007/JHEP07(2016)069
- Lee, J.S.H., Park, I., Watson, I.J. et al. Quark-Gluon Jet Discrimination Using Convolutional Neural Networks. J. Korean Phys. Soc. 74, 219–223 (2019). https://doi.org/10.3938/jkps.74.219
- Komiske, Patrick T., Eric M. Metodiev and Matthew D. Schwartz. “Deep learning in color: towards automated quark/gluon jet discrimination.” Journal of High Energy Physics 2017 (2016): n. pag.
- Albertsson, K. (2021). Machine Learning in High-Energy Physics: Displaced Event Detection and Developments in ROOT/TMVA (PhD dissertation, Luleå University of Technology). Retrieved from https://urn.kb.se/resolve?urn=urn:nbn:se:ltu:diva-87247
- Baldi, P., Sadowski, P. & Whiteson, D. Searching for exotic particles in high-energy physics with deep learning. Nat Commun 5, 4308 (2014). https://doi.org/10.1038/ncomms5308





















