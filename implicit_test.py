# import libraries
import pandas as pd
import numpy as np 
from SOM_plus_clustering.modules.som import SOM
from SOM_plus_clustering.modules.kmeans import KMeans
from SOM_plus_clustering.modules.variables import INITIATION_METHOD_LIST
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import time

def get_sc(X,y):
    try:
        return silhouette_score(X,y)
    except:
        return -1

def test_clustering_method(X: np.ndarray, total_trial: int, som_max_iteration:int, som_lr: float, som_nr:int, epoch:int, kmeans_total_cluster:int, som_m:int, som_n:int, path:str) -> pd.DataFrame:
    """
    Collects silhouette score of kmeans and SOM method with different initiator method.

    Args:
        X (np.ndarray): Values of test dataset.
        total_trial (int): Number of iteration for each clustering.
        som_max_iteration (int): Maximum iteration for self organizing matrix.
        som_lr (float): Value of learning rate (alpha) of Self Organzing Matrix.
        som_nr (int): Value of radius (gamma) of Self Organizing Matrix.
        epoch (int): Number of training iteration.
        kmeans_total_cluster (int): Total number of center that initiated in kmeans.
        som_m (int): Height of matrix in Self Organizing Matrix.
        som_n (int): Width of matrix in Self Organizing Matrix.

    Returns:
        pd.DataFrame: Table of silhouette score for each method.
    """
    list_data = []
    list_time = []
    for i in tqdm(range(total_trial)):
        list_ex_time = []
        kmeans_model2 = KMeans(n_clusters=kmeans_total_cluster, method="kmeans++")
        ts = time.time()
        kmeans_model2.fit(X, epochs=epoch)
        ts = time.time() - ts
        list_ex_time.append(ts)
        kmeans_model2_pred = kmeans_model2.predict(X)
        list_kmeans_eval = [get_sc(X, kmeans_model2_pred)]
        list_som_model_eval = []
        
        for methods in INITIATION_METHOD_LIST:
            if methods != "kde_kmeans":
                som_model = SOM(m = som_m, n = som_n, 
                                dim = X.shape[1], initiate_method = methods, 
                                max_iter = som_max_iteration, 
                                learning_rate = som_lr, neighbour_rad = som_nr, distance_function="euclidean")
                ts = time.time()
                som_model.fit(X=X, epoch=epoch)
                ts = time.time() - ts
                list_ex_time.append(ts)
                pred = som_model.predict(X=X)
                #eval_score = som_model.evaluate(X=X, method="silhouette")
                eval_score = get_sc(X, pred)
                list_som_model_eval.append(eval_score)
        print(list_kmeans_eval+list_som_model_eval)
        list_data.append(list_kmeans_eval+list_som_model_eval)
        list_time.append(list_ex_time)
        print("saving data") 
        data_table = pd.DataFrame(list_data, columns=["kmeans++", "random SOM", "kmeans SOM", "kmeans++ SOM", "SOM++", "kde SOM"])
        time_table = pd.DataFrame(list_time, columns=["kmeans++", "random SOM", "kmeans SOM", "kmeans++ SOM", "SOM++", "kde SOM"])
        data_table.to_csv(path, index=False)
        time_table.to_csv("Datas/time_execution.csv", index=False)
    return
        

if __name__ == "__main__":
    # read dataset to be tested in clustering
    df = pd.read_csv('iris_data.csv', header=None)
    df.drop(4, axis=1, inplace=True)
    df.head()
    # extract value from dataframe
    test_values = df.values

    # normalize the data
    test_values = preprocessing.normalize(test_values)

    # create a table of silhouette score
    test_clustering_method(X = test_values, 
                                    total_trial = 50, 
                                    som_max_iteration = None, 
                                    som_lr = 0.7, 
                                    som_nr = 4, 
                                    epoch = 10000, 
                                    kmeans_total_cluster = 8, 
                                    som_m = 4, 
                                    som_n = 2,
                                    path="Datas/silhouette_score_data_dummy.csv")