import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Load dataset (e.g., MNIST digits)
def load_data():
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y

# Build autoencoder model
def build_autoencoder(input_dim, latent_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(latent_dim, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return autoencoder, encoder

# Perform dimensionality reduction using autoencoder and t-SNE
def reduce_dimensions(X, latent_dim=32, tsne_dim=2):
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Build and train autoencoder
    autoencoder, encoder = build_autoencoder(X_scaled.shape[1], latent_dim)
    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=256, shuffle=True, verbose=0)
    
    # Extract latent space representation
    latent_space = encoder.predict(X_scaled)
    
    # Apply t-SNE on the latent space
    tsne = TSNE(n_components=tsne_dim, random_state=42)
    X_embedded = tsne.fit_transform(latent_space)
    
    return X_embedded

# Apply clustering on the reduced data
def apply_clustering(X_embedded, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_embedded)
    return clusters

# Plot the results
def plot_results(X_embedded, clusters, y):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Clustering in 2D Latent Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

# Main function
def main():
    # Load data
    X, y = load_data()
    
    # Reduce dimensions using autoencoder and t-SNE
    X_embedded = reduce_dimensions(X)
    
    # Apply clustering on the reduced data
    clusters = apply_clustering(X_embedded)
    
    # Plot the results
    plot_results(X_embedded, clusters, y)

if __name__ == "__main__":
    main()
