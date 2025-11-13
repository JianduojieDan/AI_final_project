# AI Final Project: A Data-Driven Model for Retail Site Selection

This project, developed for the "Intro to AI" course, implements a machine learning pipeline to predict the viability of new convenience store locations. The methodology leverages geospatial data, OpenStreetMap (OSM) points of interest, and demographic data to train a classification model.

## 1. Project Overview

The core objective is to automate and enhance the strategic decision-making process for retail site selection. The problem is framed as a **binary classification task**, where the model learns to predict whether a given location is "suitable" or "unsuitable" for a new convenience store based on a vector of engineered features.

* **Model:** Random Forest Classifier
* **Key Data Sources:** OpenStreetMap (OSM), NSW Geopackage (GDA2020)
* **Core Technologies:** Python, Pandas, GeoPandas, Scikit-learn, Osmium

## 2. Methodology

Our approach wasn't just to throw data at a model. We followed a **deliberate** data
science workflow, designed to build a **meaningful** understanding of what makes a location successful,
from the ground up.

### 2.1. Data Sourcing & Feature Engineering

To build a truly predictive model, we needed to capture two different "views" of a location: 
the micro-level, "on-the-street" reality, and the macro-level, demographic context.

* **OpenStreetMap (OSM) Data:** This is our "on-the-street" view. We parsed the massive 
* `australia-251105.osm.pbj` file to extract the features that a potential customer would actually
see and interact with. This includes:
    * **Competitor Analysis:** Proximity and density of other convenience stores or supermarkets.
    * **Traffic Drivers:** Location of transport hubs (bus stops, train stations).
    * **Points of Interest:** Density of amenities like schools, offices, and parks that draw 
  in foot traffic.

* **Geospatial & Demographic Data (NSW GDA2020):** This is our "macro" view. The `.gpkg` files 
provide the official story for the NSW regions (G01, G33, G62). From this, we engineered features 
that describe the *environment* of a location, such as population density, administrative boundaries,
and other key demographic indicators.

Neither data source is **sufficient** on its own. A high-density population area (from Geopackage)
might already be **saturated** with competitors (from OSM). Only by combining them do we get the
complete picture.

### 2.2. Data Preprocessing & Integration

This is the **challenging** (and most **crucial**) part: taking our two very different, 
very raw data sources and forging them into a single, unified dataset that a machine can actually 
learn from.

The scripts in `Data_selection/` and `script_to_clean_data/` handle this "wrangling." They parse
the raw files, standardize formats, and handle missing values.

The `joined_dataset_and_script/` directory is where it all comes together. Here, we:
1.  **Integrate:** We merge the OSM (micro) features and Geopackage (macro) features, 
aligning them to specific locations.
2.  **Define a Target:** We create our target variable (the "answer" we want the model to learn).
Based on existing store data, we label locations as either "successful" (1) or "unsuccessful" (0).

The result of this entire process is the `FINAL_TRAINING_DATASET.csv`. Every row in this file
represents one location, and every column represents one of the features we so carefully engineered,
plus our final success label.

## 3. Model Upgrade: The Journey from F1 = 0 to a Tuned Model

Our initial modeling attempts (documented in `Data_selection/` scripts) revealed a critical flaw. While the model had high *accuracy* (96%), it was useless. It achieved this by simply predicting "Not Suitable" for every location.

This section documents the step-by-step iteration, recorded in the root directory, that took our model from non-functional to fully tuned.

The core challenge was **extreme class imbalance**: our training data contained 15,179 "Not Suitable" (Class 0) locations but only 621 "Suitable" (Class 1) locations.

### 3.1 `Baseline_RF.py`

* **Problem:** We needed to establish a starting point. What happens if we do nothing special?
* **Tool:** A standard `RandomForestClassifier` with default settings.
* **Reason:** This is the "naive" attempt. We must first prove that a problem exists.
* **Result (Failure):** A 96% **accuracy** but a **0.00 F1-score** for `Class 1`. The model was completely "blind" and learned to ignore all 621 suitable locations. This confirmed the imbalance was a critical issue.

### 3.2 `Baseline_RF_with_class_weight.py` (and `RF_with_scikit-learn.py`)

* **Problem:** The naive model was "blind" (F1=0.00). We needed to force it to pay attention to the rare `Class 1`.
* **Tool:** The `class_weight='balanced'` parameter built into Scikit-learn.
* **Reason (Algorithm-Level Fix):** This is the quickest solution. It's an "algorithm-level" fix that tells the model, "You will be *penalised* 100x more for misclassifying a rare `Class 1` sample than a common `Class 0` sample."
* **Result (Failure):** This strategy failed. The F1-score only improved from 0.00 to **0.01**. This proved that simply "penalising" the model was not enough; the data itself was the problem.

### 3.3 `RF_with_SMOTE.py`

* **Problem:** The algorithm-level fix (`class_weight`) failed. We needed to fix the *data* itself.
* **Tool:** `SMOTE` (Synthetic Minority Over-sampling TEchnique), from the `imbalanced-learn` library.
* **Reason (Data-Level Fix):** Instead of punishing the model, we "fixed" the data. SMOTE analyses the 621 `Class 1` samples and generates thousands of new, synthetic, but realistic `Class 1` samples. This gave us a perfectly balanced 1:1 dataset for the model to train on.
* **Result (Success!):** This was our first major breakthrough. The F1-score jumped from 0.01 to **0.29**. This script established our first successful, working baseline. The model could finally "see" and learn the patterns of a suitable location.

### 3.4 `RF_with_SMOTE_rainforced_GridSearchCV.py`

* **Problem:** Our 0.29 F1-score was good, but it was based on default parameters. We needed to find the *optimal* parameters (e.g., `max_depth`, `n_estimators`) to build the best model possible.
* **Tool:** `GridSearchCV` combined with an `imbalanced-learn Pipeline`.
* **Reason (Professional Tuning):** This is the final, professional step.
    1.  `GridSearchCV` automatically tests hundreds of different parameter combinations (144 in our case) to find the "golden" settings.
    2.  The **`Pipeline`** was essential to fix a **"data leakage"** trap. It bundles `SMOTE` and the `RandomForestClassifier` together. This ensures that, during cross-validation, `SMOTE` is *only* applied to the training portion, not the validation portion. This prevents the model from "cheating" and gives us an honest, non-overfitted result.
* **Result (Final Model):** We found the *true* optimal model. While the final F1-score was **0.26** (an honest score), the underlying metrics were far superior. This model delivered a **Recall of 0.53**, meaning it successfully found 82 of the 155 real-world suitable locations in our test set—**more than double** the 37 locations our 0.29 baseline found. This "high-recall" model is far more valuable for finding the maximum number of business opportunities.



## 4. Author

* **Jianduojie Dan**