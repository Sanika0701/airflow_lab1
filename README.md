# airflow_lab1

# Enhanced Airflow ML Pipeline - Clustering Analysis

An advanced Apache Airflow data pipeline implementation that performs exploratory data analysis and clustering using multiple algorithms (Agglomerative Clustering and DBSCAN) with Docker containerization.

## 📋 Project Overview

This lab demonstrates a production-ready ML workflow using Apache Airflow to automate:
- **Exploratory Data Analysis (EDA)** with visualization
- **Data preprocessing** with MinMax scaling
- **Agglomerative Clustering** (hierarchical clustering)
- **DBSCAN Clustering** (density-based clustering)
- **Model evaluation** using elbow method and dendrogram analysis
- **Automated scheduling** with daily runs

  ## Airflow Output
  <img width="1898" height="866" alt="image" src="https://github.com/user-attachments/assets/17550ff7-e1f8-4a98-a094-93308c863c57" />


## 🏗️ Architecture

```
Enhanced ML Pipeline
├── EDA Task → Data Loading → Preprocessing
│                                    ├→ Agglomerative Model → Evaluation
│                                    └→ DBSCAN Model
```

## 🚀 Key Features

### 1. **Exploratory Data Analysis**
- Generates pairplot visualizations for numerical features
- Computes summary statistics
- Saves visualizations to `working_data/` directory

### 2. **Multiple Clustering Algorithms**
- **Agglomerative Clustering**: Hierarchical clustering with Ward linkage
  - Creates dendrogram visualization
  - Calculates cluster centers for prediction
  - Uses nearest-center assignment for test data
- **DBSCAN**: Density-based clustering
  - Automatically identifies clusters and noise points
  - No need to specify number of clusters upfront

### 3. **Production-Ready Features**
- Comprehensive logging at each step
- Error handling and data validation
- Automated visualization generation
- Scalable architecture with parallel task execution
- Daily automated scheduling

## 📁 Project Structure

```
airflow_lab1/
├── dags/
│   ├── src/
│   │   ├── __init__.py           # Makes src a Python package
│   │   └── lab.py                # Core ML functions
│   ├── data/
│   │   ├── file.csv              # Training dataset
│   │   └── test.csv              # Test dataset
│   ├── model/                    # Generated models (gitignored)
│   │   ├── agglomerative_model.sav
│   │   └── dbscan_model.sav
│   └── airflow.py                # DAG definition
├── logs/                         # Airflow logs (gitignored)
├── plugins/                      # Airflow plugins
├── working_data/                 # Output visualizations (gitignored)
│   ├── pairplot.png
│   ├── dendrogram.png
│   └── elbow_plot.png
├── .env                          # Environment variables (gitignored)
├── .gitignore                    # Git ignore rules
├── docker-compose.yaml           # Docker services configuration
└── README.md                     # This file
```

## 🛠️ Technologies Used

- **Apache Airflow 2.5.1** - Workflow orchestration
- **Docker & Docker Compose** - Containerization
- **Python 3.7** - Programming language
- **scikit-learn** - Machine learning algorithms
- **pandas** - Data manipulation
- **matplotlib & seaborn** - Data visualization
- **scipy** - Scientific computing (dendrogram generation)

## 📦 Prerequisites

- Docker Desktop installed and running
- At least 4GB of RAM allocated to Docker
- WSL2 (for Windows users)
- Git for version control

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd airflow_lab1
```

### 2. Create Environment File
Create a `.env` file in the root directory:
```bash
AIRFLOW_UID=50000
```

### 3. Create Required Directories
```bash
mkdir logs plugins working_data
```

### 4. Start Airflow
```bash
# Initialize the database (first time only)
docker compose up airflow-init

# Start all services
docker compose up
```

Wait until you see:
```
airflow-webserver-1 | 127.0.0.1 - - [DATE] "GET /health HTTP/1.1" 200
```

### 5. Access Airflow UI
- Open browser: `http://localhost:8080`
- Username: `airflow2`
- Password: `airflow2`

### 6. Run the Pipeline
1. Find the DAG: `Airflow_Lab1_Enhanced`
2. Toggle it **ON**
3. Click the **Play button** to trigger manually

## 📊 Pipeline Tasks

| Task | Description | Output |
|------|-------------|--------|
| `exploratory_analysis_task` | Generates EDA visualizations and statistics | `pairplot.png` |
| `load_data_task` | Loads training data from CSV | Serialized DataFrame |
| `data_preprocessing_task` | Scales features using MinMaxScaler | Normalized data |
| `build_agglomerative_model_task` | Trains hierarchical clustering model | `dendrogram.png`, `elbow_plot.png`, model file |
| `dbscan_model_task` | Trains density-based clustering model | Model file, cluster statistics |
| `evaluate_agglomerative_task` | Evaluates optimal clusters and predicts test data | Cluster assignment |

## 🔍 Key Improvements Over Original Lab

### 1. **Added EDA Task**
- Visualizes feature distributions and correlations
- Provides insights before modeling

### 2. **Multiple Clustering Algorithms**
- Compares hierarchical vs density-based approaches
- Demonstrates understanding of different clustering paradigms

### 3. **Enhanced Visualizations**
- Dendrogram for hierarchical structure
- Elbow plot for optimal cluster selection
- Pairplot for feature relationships

### 4. **Robust Prediction Logic**
- Nearest-center assignment for Agglomerative clustering
- Handles single test samples properly
- Realistic production approach

### 5. **Comprehensive Logging**
- Detailed logs at each step
- Performance metrics tracking
- Easier debugging and monitoring

### 6. **Automated Scheduling**
- Runs daily automatically
- Can be changed to `@weekly` or `None` (manual)

## 📈 Monitoring & Logs

### View Task Logs
1. Click on DAG → Graph view
2. Click on any task box
3. Select **Logs** tab

### Output Files Location
- **Visualizations**: `working_data/`
- **Models**: `dags/model/`
- **Airflow Logs**: `logs/`

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Building production ML pipelines with Airflow
- ✅ Implementing multiple clustering algorithms
- ✅ Creating comprehensive data visualizations
- ✅ Docker containerization for reproducibility
- ✅ Workflow orchestration and task dependencies
- ✅ Logging and monitoring best practices
- ✅ Handling edge cases in ML workflows

## 📄 License

This project is created for educational purposes as part of MLOps coursework.

**Note**: Remember to add sensitive files (`.env`, `logs/`, `model/`, `working_data/`) to `.gitignore` before pushing to GitHub.
