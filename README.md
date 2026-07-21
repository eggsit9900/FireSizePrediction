# Fire Size Prediction Using the Mesogeos Datacube and Machine Learning

A machine learning pipeline that classifies Mediterranean wildfire sizes (small, medium, large) from a 648GB spatio-temporal environmental dataset. It was built to run within a 12GB memory limit on Google Colab.

![Mediterranean Fire Risk Map](images/photoswap_attention.png)
*Fire risk map for Greece with attention overlay highlighting regions where soil moisture drives risk. High-risk areas concentrate in southern Greece, consistent with historical fire patterns.*

## Overview

Wildfires in the Mediterranean are intensifying due to climate change. Predicting their severity ahead of time helps fire agencies allocate resources before fires grow. This project was completed for a graduate AI course at Purdue University and uses the [Mesogeos datacube](https://orionlab.space.noa.gr/mesogeos/). It is a 648GB harmonized dataset of meteorological, vegetation, and soil moisture data covering the Mediterranean at 1km/daily resolution from 2006 to 2022. The goal is to classify expected wildfire sizes from environmental conditions.

The pipeline predicts whether a fire will be **small** (<100 ha), **medium** (100-500 ha), or **large** (>500 ha). It also generates risk maps that highlight priority regions for fire management planning. A full writeup of the methodology and results is available in [the project paper](./Mesogeos_Paper.pdf).

## Key Results

- **The 648GB datacube was processed within a 12GB memory limit** using chunked Zarr/xarray loading. Peak memory usage was about 10GB. This is an 85% reduction versus naive loading and was the core engineering challenge of the project.
- **Compute was reduced by about 90%** with a Random Forest surrogate model that maps fire risk across the full study region without exhaustive processing.
- **The end-to-end pipeline reached 72.5% classification accuracy on a synthetic benchmark.** The classifier was trained and evaluated on 200 synthetic fire events (see Limitations). This validates the pipeline mechanics rather than real-world predictive skill. Training on the real burned-area labels in Mesogeos is the next step.
- **Risk maps with attention overlays** identify high-priority regions in Greece for focused monitoring.

## How It Works

```
648GB Datacube → Chunked Loading & Subsetting → Preprocessing → Surrogate Risk Model → Fire Size Classifier → Risk Maps
     (Zarr)           (xarray, Greece region)    (imputation,      (Random Forest)      (TensorFlow NN)     (Matplotlib)
                                                   scaling)
```

**1. Memory-efficient data loading.** The datacube is loaded with xarray using optimized chunk parameters (`{'time': 20, 'x': 100, 'y': 100}`) and immediately subset to the Greek region (19.5-28.5°E, 34.5-42.0°N). This cut memory requirements by about 85% versus naive loading. At 648GB, the dataset can never be held in RAM at once.

**2. Preprocessing.** Five environmental variables known to drive fire behavior are extracted: temperature, relative humidity, wind speed, NDVI (vegetation health), and soil moisture index. About 10% of the data has missing values, mostly from cloud cover. These are mean-imputed per variable and all features are standardized.

**3. Surrogate risk modeling (DSAGE-inspired).** A Random Forest regressor adapted from the [DSAGE](https://proceedings.neurips.cc/paper_files/paper/2022/file/f649556471416b35e60ae0de7c1e3619-Paper-Conference.pdf) surrogate approach learns fire risk from a sample of locations and then predicts risk across the entire study grid. It avoids full-datacube computation while still producing region-wide risk maps.

**4. Fire size classification.** A TensorFlow neural network (64→32→3 architecture with batch normalization and dropout) classifies fire events into size categories. Feature importance analysis identified soil moisture (31.5%) and vegetation index (23.2%) as the strongest predictors. This is consistent with fire ecology.

**5. Visualization.** Risk maps use attention-style contour overlays (inspired by [Photoswap](https://proceedings.neurips.cc/paper_files/paper/2023/file/6e9a0a72da9b76c3ebc8cc33ff10ac29-Paper-Conference.pdf)) to highlight where the most important features drive risk. This keeps the maps readable for non-technical stakeholders.

## Tech Stack

Python · TensorFlow · scikit-learn · xarray · Zarr · NumPy · SciPy · Matplotlib · Google Colab

## Running the Project

1. Clone the repo:
   ```bash
   git clone https://github.com/arizzuto98/FireSizePrediction.git
   cd FireSizePrediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Mesogeos datacube from the [Mesogeos project page](https://orionlab.space.noa.gr/mesogeos/). Note that the full datacube is 648GB. The pipeline only reads chunked subsets, but you will need sufficient storage or streaming access.
4. Run the notebook end-to-end. It is designed for Google Colab Pro and falls back to CPU if TPU ops are unsupported.

Full analysis runtime is about 75 minutes on Colab Pro.

## Limitations & Next Steps

This project was built as a **proof of concept** under tight compute constraints. There are honest limitations worth knowing before interpreting the results:

- **The classifier was trained on 200 synthetic fire events** generated to match the observed 70/20/10 distribution of Mediterranean fire sizes rather than historical fire records. Because the class distribution is imbalanced, a majority-class predictor would score about 70%. The 72.5% figure should be read as a pipeline demonstration and not a measure of predictive skill. Validating against the real burned-area labels in Mesogeos is the natural next step and would give a true measure of real-world accuracy.
- **A 10-timestep temporal sample** was used to keep processing feasible. A fuller temporal range would better capture seasonal dynamics.
- **Coverage is limited to Greece.** Extending to the full Mediterranean would enable cross-regional comparison.

Planned improvements include training on real fire records, ensemble methods for the harder medium-size category, and recurrent architectures to model temporal fire risk evolution.

## References

- Kondylatos, S., Prapas, I., Camps-Valls, G., & Papoutsis, I. (2024). [Mesogeos: A multi-purpose dataset for data-driven wildfire modeling in the Mediterranean](https://proceedings.neurips.cc/paper_files/paper/2023/file/9ee3ed2dd656402f954ef9dc37e39f48-Paper-Datasets_and_Benchmarks.pdf). *NeurIPS Datasets and Benchmarks*.
- Bhatt, V., Tjanaka, B., Fontaine, M., & Nikolaidis, S. (2022). [Deep surrogate assisted generation of environments](https://proceedings.neurips.cc/paper_files/paper/2022/file/f649556471416b35e60ae0de7c1e3619-Paper-Conference.pdf). *NeurIPS 35*.
- Gu, J., Wang, Y., Zhao, N., et al. (2024). [Photoswap: Personalized subject swapping in images](https://proceedings.neurips.cc/paper_files/paper/2023/file/6e9a0a72da9b76c3ebc8cc33ff10ac29-Paper-Conference.pdf). *NeurIPS 36*.

## Author

**Anthony Rizzuto** Electrical Engineer | M.S. ECE candidate, Purdue University
