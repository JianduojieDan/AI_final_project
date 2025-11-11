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

### 2.3. Modeling & Evaluation

* **Model Choice (Random Forest):** We didn't just pick any model. We chose a 
**Random Forest Classifier** (using `RF_with_scikit_learn.py`) for several key reasons. It's 
a **robust** algorithm that excels at handling complex, real-world data with many features.
It builds hundreds of individual "decision trees" and combines their votes, which makes it
**highly accurate** and **less prone** to overfitting (i.e., memorizing the training data instead 
of learning its patterns).

* **Evaluation:** How do we know it's any good? We don't just trust the training accuracy.
We compare our main model against `Baseline_RF.py`. This baseline acts as our "sanity check." 
It's the simple model we have to beat. Our final model must **significantly outperform** this 
baseline to prove that our complex feature engineering and modeling were actually worth the effort.

## 3. Author

* **Jianduojie Dan**