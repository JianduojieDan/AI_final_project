# AI Final Project: A Data-Driven Model for Retail Site Selection

## Table of Contents
* [1. Project Overview](#1-project-overview)
* [2. Methodology](#2-methodology)
* [3. Project Roadmap & Model Iteration](#3-project-roadmap--model-iteration)
* [4. How to Run](#4-how-to-run)
* [5. Project File Structure](#5-project-file-structure)
* [6. Dependencies](#6-dependencies)
* [7. Author](#7-author)

---

## 1. Project Overview

The core objective is to automate and enhance the strategic decision-making process for retail site selection. The problem is framed as a **binary classification task**, where the model learns to predict whether a given location is "suitable" (1) or "unsuitable" (0) for a new convenience store based on a vector of engineered features.

* **Model:** Random Forest Classifier
* **Key Data Sources:** OpenStreetMap (OSM), NSW Geopackage (GDA2020)
* **Core Technologies:** Python, Pandas, GeoPandas, Scikit-learn, Imbalanced-learn，GridSearchCV, Group attributes

## 2. Methodology

My approach wasn't just to throw data at a model. We followed a deliberate data science workflow, designed to build a meaningful understanding of what makes a location successful, from the ground up.

### 2.1. Data Sourcing

To build a predictive model, we needed to capture two different "views" of a location: the micro-level, "on-the-street" reality, and the macro-level, demographic context.

* **OpenStreetMap (OSM) Data:** This is our "on-the-street" view. We parsed the `australia-251105.osm.pbf` file to extract features a potential customer would interact with (competitors, transport hubs, schools, offices, etc.).
* **Geospatial & Demographic Data (NSW GDA2020):** This is our "macro" view. The `.gpkg` files provide the official story for the NSW regions (G01 - Population, G33 - Income, G62 - Transport).

### 2.2. Data Preprocessing & Integration

This was the most crucial part: forging these raw, different data sources into a single, unified dataset.

1.  **Extraction:** Scripts in `Data_selection/` (controlled by `config.json`) parse the raw `.gpkg` files.
2.  **Merging:** `merge_features_from_G01_G33_G62.py` combines the extracted CSVs into a `MASTER_Convenience_Store_Dataset.csv`.
3.  **Spatial Join:** The script `Group_attributes_to_finalize_data.py` performs the final, critical merge. It uses a **Spatial Join** to combine our point-based store locations (`convenience_stores_locations.csv`) with the polygon-based SA1 regions. This allows us to count how many stores fall within each region, creating our target variable `store_count`.

> **What is a Spatial Join?**
>
> A spatial join is a GIS operation that merges attributes from two datasets based on their spatial relationship (such as "intersects," "contains," or "within"). In this project, we use it to "join" the latitude/longitude coordinates of stores to the SA1 census area polygons they fall inside.
>
> * **Source:** [GeoPandas Documentation on Spatial Joins](https://geopandas.org/en/stable/gallery/spatial_joins.html)
> * **Example Code:**
>     ```python
>     # 'stores_gdf' are the store points, 'sa1_main_gdf' are the area polygons
>     joined_gdf = gpd.sjoin(stores_gdf, sa1_main_gdf, how="inner", predicate="within")
>     ```

4.  **Feature Engineering:** The professor's final suggestion was implemented in `Group_attributes_to_finalize_data.py`. Instead of using all 136 raw features, this script engineers 15 *meaningful* features like `pop_density`, `core_consumer`, and `competitor_density`. This was the most important step for model performance.
## 136 columns of raw data already has been grouped to 15 features, if you want to see the origional data, please check push history
## From Iteration 1 to Iteration 5, i am using 136 columns of feature, but after iteration 5, i switched to grouped features.
The result of this entire process is the `FINAL_TRAINING_DATASET.csv`.

## 3. Project Roadmap & Model Iteration

Our final model was the result of a deliberate, multi-step iteration process, responding to failures and incorporating advice. All models share the same baseline framework: loading the dataset, creating the binary target `is_suitable_location`, and splitting into 80% train / 20% test.

### Iteration 1: `Baseline_RF.py` (The Naive Model)

* **Design:** A standard `RandomForestClassifier` with default settings, trained on the initial 136 features.
* **Problem:** The training data was extremely imbalanced (15,179 "Not Suitable" vs. 621 "Suitable").
* **Result (Poor):** While accuracy was 96%, this was misleading. The model was "blind" to the minority class, achieving a very low **F1-score of 0.13** for "Suitable" locations.
    ```
    Confusion Matrix:
           [Pred 0] [Pred 1]
    [True 0] 3790     5
    [True 1] 144      11

                          precision    recall  f1-score   support
    Class 1 (Suitable)       0.69      0.07      0.13       155
    ```

### Iteration 2: Professor's Advice (Algorithm-Level Fix)

* **Design:** After consulting with the professor, I applied the `class_weight='balanced'` parameter. This algorithm-level fix penalizes the model more for misclassifying the rare "Suitable" class.
* **Scripts:** `Baseline_RF_with_class_weight.py` and `RF_with_scikit-learn.py`.
* **Result (Failure):** This strategy failed. The F1-score dropped to **0.01**. This proved that simply "penalizing" the model was not enough; the data itself was the problem.
    ```
    Confusion Matrix:
           [Pred 0] [Pred 1]
    [True 0] 3792     3
    [True 1] 154      1

                          precision    recall  f1-score   support
    Class 1 (Suitable)       0.25      0.01      0.01       155
    ```

### Iteration 3: `RF_with_SMOTE.py` (Data-Level Fix)

* **Design:** We researched online for data-level solutions and discovered **SMOTE** (Synthetic Minority Over-sampling TEchnique). This technique synthesizes new, artificial samples of the minority class, creating a perfectly balanced 1:1 training set.
* **Source:** [imbalanced-learn Documentation on SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
* **Result (Success!):** This was our first major breakthrough. The model could finally "see" the minority class, and the F1-score jumped to **0.34**.
    ```
    Confusion Matrix:
           [Pred 0] [Pred 1]
    [True 0] 3687     108
    [True 1] 102      53

                          precision    recall  f1-score   support
    Class 1 (Suitable)       0.33      0.34      0.34       155
    ```

### Iteration 4: `RF_selector...GridSearchCV.py` (Hyperparameter Tuning)

* **Design:** An F1-score of 0.34 was a good start, but it was based on default parameters. I researched professional tuning methods and implemented `GridSearchCV` (Grid Search Cross-Validation) combined with an `imbalanced-learn Pipeline`.
* **Sources:**
    * [Scikit-learn Documentation on GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
    * [Imbalanced-learn Documentation on Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html)
* **Key Insight:** The `Pipeline` was *essential*. It bundles `SMOTE` and the `RandomForestClassifier` together, ensuring that SMOTE is *only* applied to the training folds during cross-validation, preventing data leakage and giving an honest performance estimate.

### Iteration 5: Hyperparameter Tuning (GridSearchCV Pipeline)
*Script:** `RF_selector_with_SMOTE_rainforced_GridSearchCV.py`
Design: 
Instead of guessing parameters, we built a formal pipeline integrating SMOTE and Random Forest inside a GridSearchCV.
- **Search Space**: Tested various combinations of `n_estimators` (10-500) and `max_depth` (10-30).
- **Goal**: To find the mathematically optimal configuration that balances precision and recall.

Result: 
The GridSearch identified the optimal parameters: `max_depth=3` and `n_estimators=300`. This step was crucial as it shifted our approach from "trial and error" to a "systematic search."

### Iteration 6: Professor's Advice (The "Ultimate Boost")

* **Design:** I consulted the professor again. The problem wasn't the model or the tuning—it was the **features**. We were feeding the model 136 unhelpful, raw data columns.
* **The Solution:** The `Group_attributes_to_finalize_data.py` script was created. This script implements **feature engineering** to transform the 136 raw columns into 15 highly-relevant, engineered features like `pop_density`, `core_consumer` (key age groups), `competitor_density`, and `food_density`.
* **Final Model:** We re-run the `RF_selector_with_SMOTE_rainforced_GridSearchCV.py` pipeline on these new 15 features. This final, high-recall model is far more valuable for finding the maximum number of business opportunities.

### Iteration 7: The Final Model (Robust & Generalized)
Script: `RF_after_GridsearchCV_SEARCH.py`

Design: 
We applied the best parameters found in Iteration 5 (`max_depth=3`, `n_estimators=300`) to the new, clean **15-feature binned dataset** from Iteration 6.

Result (Realistic & Actionable):
- **Recall (Suitable Class): 0.63**. The model successfully identifies 63% of all viable locations.
- **Precision: 0.08**. While precision is lower than the noisy baseline, this trade-off is intentional. In retail site selection, **Recall is King**—it is far more costly to miss a profitable location (False Negative) than to inspect a site that turns out to be bad (False Positive).
- **Comparison**: Unlike previous iterations that "memorized" noise in 136 features, this model makes decisions based on 15 interpretable, binned business logic rules.
```
Confusion Matrix:
       [Pred 0] [Pred 1]
[True 0] 2706     1089     
[True 1] 58       97       

                      precision    recall  f1-score   support
Class 1 (Suitable)       0.08      0.63      0.14       155
```
## 4. How to Run

(Instructions to test the core functionality in under 10 minutes)

-------------------------------------

Step 1: Install Dependencies

This project uses a Conda environment.

Please use the following command to create and activate the environment:

```bash
# This is the recommended method
conda create --name ai_project --file requirements.txt
conda activate ai_project
```

-------------------------------------

Step 2: Download the Project Folder

Important Data Note:

Please download only the Convenience_Store directory.
You do not need to download the large /data directory containing raw multi-GB .gpkg and .osm.pbf files.

All necessary data files — including intermediate CSVs and the FINAL_TRAINING_DATASET.csv — are already included in:

Convenience_Store/Data_for_Conven/

Therefore, you can skip all data-extraction steps and run the final models immediately.

-------------------------------------

Step 3: Test the Core Functionality (Run the Final Models)

To test the project, you can run either of the final two scripts.
Both load the pre-generated FINAL_TRAINING_DATASET.csv.

Navigate to the main script folder:

```bash
cd Convenience_Store/joint_dataset_and_script
```

-------------------------------------

Option A (Recommended): Run the full GridSearchCV pipeline

Main script:

RF_selector_with_SMOTE_rainforced_GridSearchCV.py

It loads the 15-feature dataset and runs:
- SMOTE oversampling
- GridSearchCV
- RandomForest pipeline

This is very quick.

Run it:

```bash
python RF_selector_with_SMOTE_rainforced_GridSearchCV.py
```

-------------------------------------

Option B: Run the test script with pre-found parameters

Script:

RF_after_GridsearchCV_SEARCH.py

Uses parameters:
- max_depth = 3
- n_estimators = 300

Run it:

```bash
python RF_after_GridsearchCV_SEARCH.py
```

## 5. Project File Structure
```
final_project/
│
├── Convenience_Store/
│   │
│   ├── Data_for_Conven/
│   │   ├── convenience_stores_locations.csv  (Raw store GPS points)
│   │   ├── G01.conven.csv          (Intermediate extracted population data)
│   │   ├── G33.conven.csv          (Intermediate extracted income data)
│   │   ├── G62.conven.csv          (Intermediate extracted transport data)
│   │   ├── osm_features.csv        (Intermediate extracted OSM data)
│   │   ├── MASTER_Convenience_Store_Dataset.csv (Merged 136-feature dataset)
│   │   └── FINAL_TRAINING_DATASET.csv     (FINAL 15-feature engineered dataset)
│   │
│   ├── Data_selection/
│   │   ├── config.json             (CRUCIAL: Defines which columns to extract)
│   │   ├── G01.py                  (Extractor for G01 population .gpkg)
│   │   ├── G33.py                  (Extractor for G33 income .gpkg)
│   │   ├── G62.py                  (Extractor for G62 transport .gpkg)
│   │   └── osmium_feature_counter.py (Extractor for .osm.pbf file)
│   │
│   └── joint_dataset_and_script/
│       ├── merge_features_from_G01_G33_G62.py  (Merges CSVs into MASTER_...csv)
│       ├── Group_attributes_to_finalize_data.py  (CRUCIAL: Performs Feature Engineering & Spatial Join)
│       │
│       ├── Baseline_RF.py                (Iteration 1: Naive model)
│       ├── Baseline_RF_with_class_weight.py (Iteration 2: Professor's advice #1)
│       ├── RF_with_scikit-learn.py       (Iteration 2: Variant)
│       ├── RF_with_SMOTE.py              (Iteration 3: Data-level fix)
│       ├── RF_after_GridsearchCV_SEARCH.py (Iteration 5: Test of bad params)
│       └── RF_selector_with_SMOTE_rainforced_GridSearchCV.py (Iteration 4/6: FINAL MODEL SCRIPT)
│
├── data/
│   ├── Geopackage_.../               (Raw .gpkg data folders - large files)
│   └── australia-251105.osm.pbf      (Raw OSM data - very large file)
│
├── script_to_clean_data/             (Help scripts to explore raw data)
│   ├── get_name/                   (G01_header, G33_header...)
│   └── ...
│
├── README.md                         (This file)
└── requirements.txt                  (Python environment dependencies)
```
## 6. Dependencies
This project was built using Python 3.10 on an osx-arm64 (Apple Silicon) platform. Key libraries are listed in requirements.txt.

Core Libraries:

pandas & geopandas

fiona & pyogrio (For reading .gpkg files)

osmium (For reading .osm.pbf files)

scikit-learn

imbalanced-learn (For SMOTE)

## 7. Author
Jianduojie Dan
