🎧 Spotify Data Analysis Pipeline
---

📌 Overview  
This project delivers an end-to-end data analysis pipeline for exploring Spotify music data. It automates data ingestion, preprocessing, exploratory data analysis (EDA), and baseline machine learning to uncover patterns behind song popularity, audio characteristics, and genre trends.

The system integrates track-level metadata with audio feature data to generate clear, reproducible insights into listener preferences and music characteristics.

---

🚀 Key Features  

🔄 Automated Data Pipeline  
Structured ingestion and preprocessing of Spotify track and audio feature datasets from CSV sources.

📊 Exploratory Data Analysis (EDA)  
Comprehensive analysis covering:
- Song popularity trends  
- Audio feature relationships (energy, danceability, loudness)  
- Genre-level statistics  
- Temporal trends across release years  

🤖 Baseline Machine Learning  
Optional Linear Regression or Random Forest model to predict song popularity using audio features.

📈 High-Quality Visualisations  
Automatically generated correlation heatmaps, regression plots, bar charts, and distributions using Matplotlib and Seaborn.

💾 Automated Outputs  
All plots and analytical reports are saved to a structured outputs directory for reproducibility.

🛡️ Validation & Error Handling  
Built-in checks for missing files, required columns, and safe execution of the pipeline.

---

🧠 System Architecture  

The pipeline is composed of modular components:

- **data_loader.py**  
  Handles data ingestion, validation, and preprocessing of Spotify datasets.

- **analysis.py**  
  Performs exploratory data analysis and statistical summarisation.

- **modeling.py**  
  Trains and evaluates baseline machine learning models for popularity prediction.

- **visualization.py**  
  Generates and styles all analytical plots and visual outputs.

- **output_manager.py**  
  Manages output directories and saves reports, plots, and run manifests.

- **main.py**  
  Orchestrates the full pipeline via configurable execution options.

---

🛠️ Getting Started  

### Prerequisites
- 🐍 Python 3.8+  
- 📦 Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

---

📥 Installation  

Clone the repository:
- bash
- git clone https://github.com/yourusername/spotify-data-analysis-pipeline.git
- cd spotify-data-analysis-pipeline


- Install dependencies:
pip install -r requirements.txt
Place your datasets (dataset.csv and SpotifyFeatures.csv) in the data/ directory or specify paths via the CLI

---

▶️ Usage

- Run the full analysis pipeline:
  python main.py

- Use custom dataset paths:
  python main.py --tracks /path/to/tracks.csv --features /path/to/features.csv

- Skip machine learning:
  python main.py --skip-ml

- Select a model:
  python main.py --model linear_regression

- Specify a custom output directory:
  python main.py --output ./my_results
  
---

📤 Outputs

After execution, the output directory will contain:

📊 Plots
- Correlation heatmaps
- Regression plots
- Genre popularity charts

📄 Reports
- Summary statistics
- Top and least popular tracks
- Genre-level metrics
- Model performance (RMSE, R²)
  
🧾 Run Manifest
- Execution timestam
- Generated outputs overview
  
---

📄 License
This project is intended for educational and portfolio purposes.

---


