from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_optimal_clusters(X_train, max_clusters=10):
    # Use the Elbow Method to find the optimal number of clusters.
    inertia = []
    for n in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(X_train)
        inertia.append(kmeans.inertia_)
    
    plt.plot(range(1, max_clusters+1), inertia, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

def train_clustering(X_train, n_clusters=3):
    # Build a clustering model using KMeans.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)
    return kmeans

def evaluate_clustering(model, X_test):
    # Predict clusters on the test set.
    labels = model.predict(X_test)
    return labels
