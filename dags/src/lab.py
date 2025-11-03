import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from kneed import KneeLocator
import pickle
import os
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

def exploratory_analysis():
    """
    Performs exploratory data analysis on the dataset.
    Creates visualizations and returns summary statistics.
    
    Returns:
        dict: Summary statistics of the dataset.
    """
    logging.info("Starting exploratory data analysis...")
    
    # Load data
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    
    # Generate summary statistics
    summary = df.describe().to_dict()
    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Summary statistics: {summary}")
    
    # Create pairplot for numerical columns
    numeric_df = df.select_dtypes(include='number')
    sns.pairplot(numeric_df)
    
    # Save to working_data directory
    output_path = '/opt/airflow/working_data/pairplot.png'
    plt.savefig(output_path)
    plt.close()
    
    logging.info(f"Pairplot saved to {output_path}")
    
    return summary


def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.

    Returns:
        bytes: Serialized data.
    """
    logging.info("Loading data from CSV...")
    
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)
    
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    
    return serialized_data
    

def data_preprocessing(data):
    """
    Deserializes data, performs data preprocessing, and returns serialized clustered data.

    Args:
        data (bytes): Serialized data to be deserialized and processed.

    Returns:
        bytes: Serialized clustered data.
    """
    logging.info("Starting data preprocessing...")
    
    df = pickle.loads(data)
    df = df.dropna()
    
    logging.info(f"Rows after dropping NaN: {len(df)}")
    
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    
    logging.info("Data preprocessing completed successfully")
    
    return clustering_serialized_data


def build_save_model(data, filename):
    """
    Builds an Agglomerative Clustering model, saves it to a file, and returns distance scores.
    Also creates and saves a dendrogram and cluster centers.

    Args:
        data (bytes): Serialized data for clustering.
        filename (str): Name of the file to save the clustering model.

    Returns:
        tuple: (distance_scores, cluster_centers, fitted_model_data)
    """
    logging.info("Building Agglomerative Clustering model...")
    
    df = pickle.loads(data)
    
    # Calculate linkage for dendrogram
    linkage_matrix = linkage(df, method='ward')
    
    # Create dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    # Save dendrogram
    dendrogram_path = '/opt/airflow/working_data/dendrogram.png'
    plt.savefig(dendrogram_path)
    plt.close()
    
    logging.info(f"Dendrogram saved to {dendrogram_path}")
    
    # Calculate distance scores for different numbers of clusters
    distance_scores = []
    
    for k in range(2, 20):  # Agglomerative needs at least 2 clusters
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        model.fit(df)
        
        # Store the distance from linkage matrix for this k
        if k < len(linkage_matrix):
            distance_scores.append(linkage_matrix[-(k-1), 2])
    
    logging.info(f"Distance scores computed for k=2 to k=19")
    
    # Create elbow plot for distances
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(distance_scores) + 2), distance_scores, marker='o')
    plt.title('Elbow Curve for Agglomerative Clustering')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distance Threshold')
    plt.grid(True)
    
    # Save elbow plot
    elbow_plot_path = '/opt/airflow/working_data/elbow_plot.png'
    plt.savefig(elbow_plot_path)
    plt.close()
    
    logging.info(f"Elbow plot saved to {elbow_plot_path}")
    
    # Build final model with optimal clusters (let's use 5 as default)
    optimal_k = 5
    final_model = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    cluster_labels = final_model.fit_predict(df)
    
    # Calculate cluster centers (mean of points in each cluster)
    cluster_centers = []
    for i in range(optimal_k):
        cluster_points = df[cluster_labels == i]
        center = cluster_points.mean(axis=0)
        cluster_centers.append(center)
    
    cluster_centers = pd.DataFrame(cluster_centers)
    logging.info(f"Calculated {len(cluster_centers)} cluster centers")
    
    # Create model directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)

    # Save the model data (model object, training data, and cluster centers)
    model_data = {
        'model': final_model,
        'training_data': df,
        'cluster_centers': cluster_centers,
        'n_clusters': optimal_k
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logging.info(f"Agglomerative model saved to {output_path}")
    
    # Return both distance scores and serialized cluster centers for XCom
    return pickle.dumps((distance_scores, cluster_centers.values, df))


def build_save_dbscan_model(data_serialized, filename):
    """
    Builds a DBSCAN clustering model and saves it to a file.

    Args:
        data_serialized (bytes): Serialized preprocessed data for clustering.
        filename (str): Name of the file to save the DBSCAN model.

    Returns:
        str: String representation of cluster labels.
    """
    logging.info("Building DBSCAN clustering model...")
    
    data = pickle.loads(data_serialized)
    
    # Build DBSCAN model
    model = DBSCAN(eps=0.5, min_samples=5).fit(data)
    
    # Count clusters (excluding noise points labeled as -1)
    n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    n_noise = list(model.labels_).count(-1)
    
    logging.info(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
    
    # Create model directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    
    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    logging.info(f"DBSCAN model saved to {output_path}")
    
    return str(model.labels_.tolist())


def load_model_elbow(filename, model_output):
    """
    Loads a saved Agglomerative clustering model and assigns test data to nearest cluster.
    This approach works even with single test samples.

    Args:
        filename (str): Name of the file containing the saved clustering model.
        model_output (bytes): Serialized tuple of (distance_scores, cluster_centers, training_data)

    Returns:
        int: The predicted cluster label for test data.
    """
    logging.info("Loading model and analyzing optimal clusters...")
    
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    
    # Load the saved model data
    model_data = pickle.load(open(output_path, 'rb'))
    
    # Unpack model output from XCom
    distance_scores, cluster_centers, training_data = pickle.loads(model_output)
    
    # Load test data
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    
    # Use KneeLocator to find optimal clusters
    kl = KneeLocator(
        range(2, len(distance_scores) + 2), 
        distance_scores, 
        curve="convex", 
        direction="decreasing"
    )

    # Optimal clusters
    optimal_clusters = kl.elbow if kl.elbow else 5
    logging.info(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Optimal no. of clusters: {optimal_clusters}")

    # Preprocess test data the same way as training data
    test_scaler = MinMaxScaler()
    test_preprocessed = test_scaler.fit_transform(df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]])
    
    # For each test sample, find the nearest cluster center
    from scipy.spatial.distance import cdist
    
    # Calculate distances from test sample to all cluster centers
    distances = cdist(test_preprocessed, cluster_centers, metric='euclidean')
    
    # Assign to nearest cluster
    predicted_cluster = distances.argmin(axis=1)[0]
    
    logging.info(f"Test sample assigned to cluster {predicted_cluster} based on nearest center")
    logging.info(f"Distance to assigned cluster: {distances[0][predicted_cluster]:.4f}")
    print(f"Test data assigned to cluster: {predicted_cluster}")
    
    return int(predicted_cluster)