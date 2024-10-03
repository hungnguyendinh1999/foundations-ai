import numpy as np

from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances , _euclidean_distances
import numbers

#################################################################
# Load Dataset
#################################################################

dataset = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    subset="all",
    shuffle=True,
    random_state=42,
)

labels = dataset.target
unique_labels, category_sizes = np.unique(labels, return_counts=True)
true_k = unique_labels.shape[0]

#################################################################
# Evaluate Fitness
#################################################################
def fit_and_evaluate(km, X, n_runs=5):

    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        km.fit(X)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
        scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
        scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, km.labels_)
        )
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")

#################################################################
# Vectorize 
#################################################################
vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5,
    stop_words="english",
)

X_tfidf = vectorizer.fit_transform(dataset.data)

#################################################################
# (TODO): Implement K-Means  
#################################################################

def check_random_state(seed):
    # check_random_state from sklearn.utils
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)

class KMeans:
    labels_ = [] # predicted labels (y)
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        

    def fit(self, X_train):
        n_samples, n_features = X_train.shape
        self.centroids = np.empty((self.n_clusters, n_features), dtype=X_train.dtype)

        # Randomly select centroid
        random_state = check_random_state(self.random_state)
        center_id = random_state.choice(n_samples) # return an integer
        self.centroids[0] = X_train[center_id].toarray() 

        # Initialize K (n_clusters) centroids #
        for c in range(1, self.n_clusters):
            # Calculate distances from points to the centroids
            dists = np.sum(euclidean_distances(self.centroids, X_train), axis=0)
            dists = dists/np.sum(dists) # Normalize the distances
            # Choose remaining points based on their distances
            new_centroid_idx = random_state.choice(n_samples, size=1, p=dists)
            self.centroids[c] = X_train[new_centroid_idx].toarray()
            
        # Iterate, adjust centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and (iteration < self.max_iter):
            # Assigning vector to nearest centroid #
            dists = euclidean_distances(self.centroids, X_train)
            centroid_idx = np.argmin(dists, axis=0)
                
            # Save current centroids
            prev_centroids = self.centroids
            
            # Update centroids by mean of cluster #
            new_centroids = np.empty((self.n_clusters, n_features), dtype=X_train.dtype)
            for i in range(self.n_clusters):
                cluster_i = X_train[centroid_idx == i]
                if cluster_i.size == 0: 
                    new_centroids[i] = prev_centroids[i]
                else:
                    new_centroids[i] = np.mean(cluster_i, axis=0)

            # Reassign centroids to new centroids
            self.centroids = new_centroids

            iteration += 1
        
        # Now that's done, let's calculate the current labels
        distances = euclidean_distances(self.centroids, X_train)
        self.labels_ = np.argmin(distances, axis=0)
        

    def set_params(self, random_state):
        self.random_state = random_state


kmeans = KMeans(n_clusters=true_k, max_iter=100)
# Feel free to change the number of runs
fit_and_evaluate(kmeans, X_tfidf)
