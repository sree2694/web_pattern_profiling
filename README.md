
---

# Web Pattern Navigation Profiling - AI Project

## Overview
This project leverages Artificial Intelligence (AI) to analyze user browsing behavior, extract patterns, and predict socio-demographic characteristics based on browsing sequences. The goal is to extract meaningful insights from user navigation patterns to understand preferences and predict various attributes like age group and gender.

## Project Phases
The project is divided into several phases, each targeting a specific aspect of the AI model development:

### 1. **Data Simulation**:
- **Description**: Generates synthetic user browsing data, simulating real-world browsing behavior.
- **Tools Used**: Python (Pandas, Random Data Generation)

### 2. **Pattern Mining**:
- **Description**: Extracts frequent n-gram patterns from the user's browsing sequence, identifying common navigation routes.
- **Tools Used**: Python (NLTK, Regular Expressions)

### 3. **Statistical Analysis**:
- **Description**: Applies **Bonferroni** and **FDR corrections** to analyze demographic significance based on extracted patterns.
- **Tools Used**: Python (SciPy, Pandas)

### 4. **Unsupervised Learning (Clustering)**:
- **Description**: Uses clustering algorithms to group similar browsing patterns and identify user segments.
- **Tools Used**: Python (KMeans, DBSCAN, Scikit-learn)

### 5. **Supervised Learning (Prediction)**:
- **Description**: Predicts socio-demographic attributes like **age group** or **gender** using features derived from browsing patterns.
- **Tools Used**: Python (RandomForest, Train-Test Split, Cross-validation)

### 6. **Feature Engineering**:
- **Description**: Extracts advanced features from browsing sequences, such as **sequence length**, **unique domains**, and **domain repetition**.
- **Tools Used**: Python (Pandas, Feature Extraction)

### 7. **Cross-validation**:
- **Description**: Performs **5-fold cross-validation** to evaluate the robustness and generalization of the AI model.
- **Tools Used**: Python (Scikit-learn, StratifiedKFold)

### 8. **Interface (GUI/CLI)**:
- **Description**: Provides both a command-line and graphical user interface to allow users to run the model in **unsupervised** or **supervised** mode.
- **Tools Used**: Python (Tkinter, Argparse)

### 9. **Logging/Exporting Results**:
- **Description**: Logs the results and exports them to CSV files for further analysis.
- **Tools Used**: Python (CSV)

## Technologies Used
- **Python 3.x**
- **Libraries**: 
  - **Scikit-learn** (for machine learning models)
  - **Pandas** (for data manipulation)
  - **NumPy** (for mathematical operations)
  - **SciPy** (for statistical analysis)
  - **Tkinter** (for GUI)

## Project Structure
```bash
├── data_simulation.py      # Data generation and preprocessing
├── pattern_mining.py       # Pattern extraction from browsing sequences
├── statistical_analysis.py # Statistical analysis and hypothesis testing
├── supervised_model.py     # Supervised learning and model evaluation
├── feature_engineering.py  # Feature extraction and advanced feature engineering
├── unsupervised_main.py    # Unsupervised learning and clustering
├── main.py                 # Entry point with UI for supervised/unsupervised modes
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation (this file)
```

## How to Run the Project

### 1. Clone the repository:
```bash
git clone https://github.com/sree2694/web_pattern_profiling.git
cd web_pattern_profiling
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the project:
- **CLI Mode**:
```bash
python main.py
```

## Contributions
Feel free to fork and contribute to this project. If you have any questions or suggestions, please open an issue or submit a pull request.

## License
This project is licensed under the Apache License.

---

Let me know if you need more customization or adjustments!