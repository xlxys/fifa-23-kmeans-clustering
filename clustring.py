import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from IPython.display import clear_output

class App(tk.Tk):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("K-Means Clustering GUI")
        
        # File selection frame
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10)
        
        self.file_label = tk.Label(file_frame, text="Select a dataset:")
        self.file_label.pack(side="left")
        
        self.file_button = tk.Button(file_frame, text="Browse", command=self.browse_file)
        self.file_button.pack(side="left", padx=10)
        
        # Dataset info frame
        info_frame = tk.Frame(self.root)
        info_frame.pack(pady=10)

        # Dataset name label
        self.dataset_name_label = tk.Label(info_frame, text="")
        self.dataset_name_label.pack()

        # Number of rows label
        self.num_rows_label = tk.Label(info_frame, text="")
        self.num_rows_label.pack()

        # Column names label
        self.col_names_label = tk.Label(info_frame, text="")
        self.col_names_label.pack()

        # Column selection frame
        col_frame = tk.Frame(self.root)
        col_frame.pack(pady=10)

        self.col_label = tk.Label(col_frame, text="Select columns to use:")
        self.col_label.pack(side="left")

        # Create a listbox widget for column selection
        self.col_listbox = tk.Listbox(col_frame, selectmode="multiple", width=50)
        self.col_listbox.pack(side="left", padx=10)

        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.run_button = tk.Button(button_frame, text="Run K-Means Clustering", command=self.run_kmeans)
        self.run_button.pack()
        
        # Optimal K label
        self.optimal_k_label = tk.Label(self.root, text="")
        self.optimal_k_label.pack()
        
        # Cluster plot frame
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack()
        
        self.root.mainloop()
        
    def browse_file(self):
        self.filename = filedialog.askopenfilename()
        
        # Update dataset info labels
        dataset_name = self.get_dataset_name()
        num_rows = self.get_num_rows()
        col_names = self.get_col_names()

        self.dataset_name_label.config(text=f"Dataset Name: {dataset_name}")
        self.num_rows_label.config(text=f"Number of Rows: {num_rows}")

        # Clear previous items from the listbox
        self.col_listbox.delete(0, tk.END)

        # Add column names to the listbox
        for col in self.columns:
            self.col_listbox.insert(tk.END, col)
        
    def get_dataset_name(self):
        return self.filename.split("/")[-1]
    
    def get_num_rows(self):
        if hasattr(self, "filename"):
            df = pd.read_csv(self.filename)
            return df.shape[0]
        else:
            return "N/A"

    def get_col_names(self):
        if hasattr(self, "filename"):
            df = pd.read_csv(self.filename)
            self.columns = df.columns.tolist()
            return ", ".join(self.columns)
        else:
            return "N/A"
        
    def run_kmeans(self):

        if not hasattr(self, "filename"):
            tk.messagebox.showerror("Error", "Please select a dataset.")
            return
        
        selected_cols = [self.col_listbox.get(idx) for idx in self.col_listbox.curselection()]
        print("Selected columns:", selected_cols)
        
        df = pd.read_csv(self.filename)
        df = df.dropna(subset=selected_cols)
        
        # onehot encoding nominal columns
        for col in selected_cols:
            if df[col].dtype == "object":
                dummies = pd.get_dummies(df[col] , prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(col, axis=1)
                selected_cols.remove(col)
                selected_cols.extend(dummies.columns)
                
        
        data = df[selected_cols].copy()
        data = ((data - data.min()) / (data.max() - data.min())) * 10 + 1
        
        optimal_k = self.gap_statistic(data)
        print("Optimal K:", optimal_k)
        self.optimal_k_label.config(text=f"Optimal K: {optimal_k}")

        max_iterations = 100   

        # Assign labels randomly to clusters
        labels = self.random_labels(data, optimal_k)

        centroids = self.random_centroids(data, optimal_k)
        old_centroids = pd.DataFrame()
        iteration = 1

        while iteration < max_iterations and not centroids.equals(old_centroids):
            old_centroids = centroids
            
            labels = self.get_labels(data, centroids)
            centroids = self.new_centroids(data, labels, optimal_k)
            self.plot_clusters(data, labels, centroids, iteration)
            iteration += 1
        
    
 

    def gap_statistic(self,X):
        # Fit KMeans clustering for different values of k
        ks = range(1, 10)
        kmeans = [KMeans(n_clusters=k).fit(X) for k in ks]

        # Get intra-cluster distances for each k
        intra_distances = [k.inertia_ for k in kmeans]

        # Generate reference datasets for each k
        ref_intra_distances = []
        for k in ks:
            # Generate random reference dataset with same shape as X
            X_ref = np.random.rand(*X.shape)

            # Fit KMeans clustering on reference dataset
            kmeans_ref = KMeans(n_clusters=k).fit(X_ref)

            # Get intra-cluster distances for reference dataset
            ref_intra_distances.append(kmeans_ref.inertia_)

        # Calculate gap statistic
        gap = np.log(ref_intra_distances) - np.log(intra_distances)
        optimal_k = np.argmax(gap) + 1

        # Train neural network to predict optimal k
        nn = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=10000)
        nn.fit(np.array(ks).reshape(-1, 1), gap)

        # Predict optimal k for new dataset
        pred_gap = nn.predict(np.array([optimal_k + 1]).reshape(-1, 1))
        optimal_k_new = int(round(optimal_k + pred_gap[0]))

        return optimal_k_new

    def random_labels(self, data, k):
        return np.random.randint(0, k, size=len(data))

    def random_centroids(self,data, k):
        centroids = []
        for i in range(k):
            centroid = data.apply(lambda x: float(x.sample()))
            centroids.append(centroid)
        return pd.concat(centroids, axis=1)

    def get_labels(self,data, centroids):
        distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
        return distances.idxmin(axis=1)

    def new_centroids(self,data, labels, k):
        centroids = data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
        return centroids

    def plot_clusters(self,data, labels, centroids, iteration):
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        centroids_2d = pca.transform(centroids.T)
        # clear_output(wait=True)
        plt.title(f'Iteration {iteration}')
        plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
        plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
        plt.show()


  


if __name__ == "__main__":
    app = App()
    app.mainloop()
