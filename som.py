"""
    References :
    Chaudhary, V., Bhatia, R. S., & Ahlawat, A. K. (2014). A novel self-organizing map (SOM) learning algorithm with nearest and farthest neurons. Alexandria Engineering Journal, 53(4), 827-831. https://doi.org/10.1016/j.aej.2014.09.007
"""

import numpy as np
import pandas as pd
import math 
import random

# Initiate random number for grid in matrix with dimension of x
def random_initiate(dim: int, min_val:float, max_val:float):
    """Initiate random number of value in range (min_val, max_val)

    Args:
        dim (int): dimension of the data
        min_val (float): minimum value of data
        max_val (float): maximum value of data

    Returns:
        np.array: array of randomly generated number
        
    Overall Complexity: O(dim)
    """
    x = [random.uniform(min_val,max_val) for i in range(dim)]
    return x

# Euclidean distance function
def euc_distance(x: np.array, y: np.array):
    """Calculate the euclidean distance of array x and y

    Args:
        x (np.array): array 1 input
        y (np.array): array 2 input

    Raises:
        ValueError: length of x and y is different

    Returns:
        float(): euclidean distance of x and y
    
    Overall Time Complexity: O(dim)
    """
    if len(x) != len(y):
        raise ValueError("input value has different length")
    else :
        dist = sum([(i2-i1)**2 for i1, i2 in zip(x, y)])**0.5
        return dist
    
def gauss(x):
    return math.exp(-0.5 * x * x)/math.sqrt(2*math.pi)

def std_dev(x):
    mean = np.mean(x)
    sums = sum( [(i - mean)**2 for i in x])**0.5
    return sums/len(x)

def kernel_gauss(x, xi):
    # silvermans bandwidth estimator
    iqr = (np.percentile(xi, 75) - np.percentile(xi, 25))/1.34
    h = iqr * (len(xi)**(-.2))
    return sum([gauss((x-i)/h) for i in xi]) / (len(xi)*h)

def deriv(x, h, xi):
    f_x = kernel_gauss(x, xi)
    f_xh = kernel_gauss(x+h, xi)
    return (f_xh-f_x)/h

def find_initial_centroid(X : np.ndarray, treshold:float):
    X = np.transpose(X)
    points = list()
    for items in X:
        xi = items
        x = np.arange(min(xi),max(xi),.001)
        y = [deriv(i, 0.001, xi) for i in x]
        local_max = list()
        for i in range(len(y)):
            if y[i] > 0 and y[i+1] < 0:
                local_max.append(i*0.001+min(xi))
        points.append(local_max)
    return points

def create_initial_centroid(X: np.ndarray, treshold, k):
    c = find_initial_centroid(X, treshold)
    new_c = np.full(shape=(k,X.shape[1]), fill_value = None)
    for i in range(k):
        for j in range(X.shape[1]):
            try: 
                new_c[i][j] = c[j][i]
            except:
                new_c[i][j] = random.uniform(np.min(X),np.max(X))
    return new_c

class kmeans():
    def __init__(self, n_clusters: int, method:str):
        self.n_clusters = n_clusters
        self.centroids = None
        self._trained = False
        self.method = method
    
    def init_centroids(self, X: np.ndarray):
        if self.method == "random":
            centroids = [random_initiate(dim=X.shape[1], min_val=X.min(), max_val=X.max()) for i in range(self.n_clusters)]
            self.centroids = centroids
        elif self.method == "kde": 
            centroids = create_initial_centroid(X, 0.001, self.n_clusters)
            self.centroids = centroids
        else:
            raise ValueError("There is no method named {}".format())
        return 
    
    def update_centroids(self, x:np.array):
        # new_centroids = np.array([X[self.cluster_labels == i].mean(axis=0) for i in range(self.k)])
        new_centroids = list()
        
        # find the distance of centroids for each data
        centroids_distance = [euc_distance(x, i) for i in self.centroids]
        
        # find the closest centroid in self.centroids
        closest_centroids_index = np.argmin(centroids_distance)
        closest_centroids = self.centroids[closest_centroids_index]
        
        # update the closest centroids to the data
        closest_centroids = [(i+j)/2 for i,j in zip(x,closest_centroids)]
        
        # update the centroid in model
        self.centroids[closest_centroids_index] = closest_centroids
        
    
    def fit(self, X: np.ndarray, epochs=3000, shuffle=True):
        if self._trained:
            raise SyntaxError("Cannot fit the model that have been trained")
        
        self.init_centroids(X)
        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(X)
            
            for x in X:
                self.update_centroids(x)
    
    def predict(self, X : np.ndarray):
        return [np.argmin([euc_distance(x, centers) for centers in self.centroids]) for x in X]

# Self Organizing Matrix Class
class SOM(): 
    def __init__(self, m: int, n: int, dim: int, initiate_method:str, max_iter: int, learning_rate:float, neighbour_rad: int) -> None:
        """_summary_

        Args:
            m (int): _description_
            n (int): _description_
            dim (int): _description_
            method (str): _description_
            max_iter (int): _description_
            learning_rate (float): _description_
            neighbour_rad (int): _description_

        Raises:
            ValueError: _description_
        
        Overall Time Complexity: O(1)
        """
        if learning_rate > 1.76:
            raise ValueError("Learning rate should be less than 1.76")
        method_type = ["random", "kmeans", "kde_kmeans"]
        if initiate_method not in method_type:
            raise ValueError("There is no method called {}".format(initiate_method))
        
        # initiate all the attributes
        self.m = m
        self.n = n
        self.dim = dim
        self.max_iter = max_iter
        self.shape = (m,n)
        self.cur_learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self._trained = False
        self.method = initiate_method
        self.cur_neighbour_rad = neighbour_rad
        self.initial_neighbour_rad = neighbour_rad
        self.neurons = None 
        self.initial_neurons = None
        
        
    
    def initiate_neuron(self, data: np.ndarray, min_val:float, max_val:float):
        """Initiate initial value of the neurons

        Args:
            min_val (float): the minimum value of the data input
            max_val (float): maximum value of the data input

        Raises:
            ValueError: There are no method named self.method or the method is not available yet

        Returns:
            list(): list of neurons to be initiate in self.neurons and self.initial_neurons
            
        Overall Time Complexity:
            self.method == "random": O(m * n * dim)
            self.method == "kmeans":
            self.method == "kmeans++":
        """
        if self.method == "random" :
            # number of step = self.dim * self.m * self.n --> O(m * n * dim)
            return [[random_initiate(self.dim ,min_val=min_val, max_val=max_val) for j in range(self.m)] for i in range(self.n)]
        elif self.method == "kmeans":
            model = kmeans(n_clusters = (self.m * self.n), method="random")
            model.fit(X = data)
            neurons = model.centroids
            neurons = np.sort(neurons, axis=0)
            neurons = np.reshape(neurons, (self.m, self.n, self.dim))
            return neurons
        elif self.method == "kde_kmeans":
            model = kmeans(n_clusters = (self.m * self.n), method="kde")
            model.fit(X = data)
            neurons = model.centroids
            neurons = np.sort(neurons, axis=0)
            neurons = np.reshape(neurons, (self.m, self.n, self.dim))
            return neurons
        else:
            raise ValueError("There is no method named {}".format(self.method))
    
    def index_bmu(self, x: np.array):
        """Find the index of best matching unit among all of neurons inside the matrix based on its euclidean distance

        Args:
            x (np.array): input array as comparison parameter

        Returns:
            tuple(): set of coordinates the best matching unit in (x,y) 
        
        Overall Time Complexity: O(m * n * dim)
        """
        neurons = np.reshape(self.neurons, (-1, self.dim)) # O(1)
        min_index = np.argmin([euc_distance(neuron, x) for neuron in neurons]) # O(m * n * dim) 
        return np.unravel_index(min_index, (self.m, self.n)) # O(1)
    
    def gaussian_neighbourhood(self, x1, y1, x2, y2):
        """Represents gaussian function as the hyper parameter of updating weight of neurons

        Args:
            x1 (_type_): x coordinates of best matching unit
            y1 (_type_): y coordinates of best matching unit
            x2 (_type_): x coordinates of the neuron
            y2 (_type_): y coordinates of the neuron

        Returns:
            float(): return the evaluation of h(t) = a(t) * exp(-||r_c - r_i||^2/(2 * o(t)^2))
        
        Overall Time Complexity: O(1)
        """
        lr = self.cur_learning_rate
        nr = self.cur_neighbour_rad
        dist = float(euc_distance([x1, y1], [x2,y2]))
        exp = math.exp(-0.5 * ((dist/nr*dist/nr)))
        val = np.float64(lr * exp)
        return val
    
    def update_neuron(self, x:np.array):
        """Update neurons based on the input data in each iteration

        Args:
            x (np.array): the input value from data
            
        Overall Complexity: O(m * n * dim)
        """
        # find index for the best matching unit index --> O(m * n * dim)
        bmu_index = self.index_bmu(x)
        col_bmu = bmu_index[0]
        row_bmu = bmu_index[1]
        
        # iterates through the matrix --> O(m * n * dim)
        for cur_col in range(len(self.neurons)):
            for cur_row in range(len(self.neurons[0])):
                # initiate the current weight of the neurons
                cur_weight = self.neurons[cur_col][cur_row]
                
                # calculate the new weight, update only if the weight is > 0
                h = self.gaussian_neighbourhood(col_bmu, row_bmu, cur_col, cur_row)
                if h > 0:
                    new_weight = cur_weight +  h * (x - cur_weight)
                    # update the weight
                    self.neurons[cur_col][cur_row] = new_weight
    
    def fit(self, X : np.ndarray, epoch : int, shuffle=True):
        """Tune the neurons to learn the data

        Args:
            X (np.ndarray): Input data
            epoch (int): number of training iteration 
            shuffle (bool, optional): the initate data to be evaluate in the matrix. 
                Defaults to True.

        Raises:
            SyntaxError: SOM._trained() already true, which the model have been trained
            ValueError: The length of data columns is different with the length of the dimension
        
        Return:
            None: fit the neurons to the data
            
        Overall Time Complexity: O(epoch * N * m * n * dim)
        """
        if self._trained:
            raise SyntaxError("Cannot fit the model that have been trained")
        
        if X.shape[1] != self.dim :
            raise ValueError("X.shape[1] should be the same as self.dim, but found {}".format(X.shape[1]))
        
        # initiate new neurons
        self.neurons = self.initiate_neuron(data=X, min_val= X.min(), max_val= X.max()) # O(m * n * dim)
        self.initial_neurons = self.neurons
        
        # initiate parameters
        global_iter_counter = 0
        n_sample = X.shape[0]
        total_iteration = min(epoch * n_sample, self.max_iter)
        
        # iterates through epoch --> O(epoch * N * m * n * dim)
        for i in range(epoch):
            if global_iter_counter > self.max_iter :
                break
            
            # shuffle the data
            if shuffle:
                np.random.shuffle(X)
            
            # iterates through data --> O(N * m * n * dim)
            for idx in X:
                if global_iter_counter > self.max_iter :
                    break
                input = idx
                
                # update the neurons --> O(m * n * dim)
                self.update_neuron(input)
                
                # update parameter and hyperparameters --> O(1)
                global_iter_counter += 1
                power = global_iter_counter/total_iteration
                self.cur_learning_rate = self.initial_learning_rate**(1-power) * math.exp(-1 * global_iter_counter/self.initial_learning_rate)
                self.cur_neighbour_rad = self.initial_neighbour_rad**(1-power) * math.exp(-1 * global_iter_counter/self.initial_neighbour_rad)
        
        self._trained = True
        
        return 
    
    def predict(self, X: np.ndarray) :
        """Label the data based on the neurons using best matching unit

        Args:
            X (np.ndarray): input data to be predicted

        Raises:
            NotImplementedError: The model have not been trained yet, call SOM.fit() first

        Returns:
            np.array(): array of the label of each data
            
        Overall Time Complexity: O(N * m * n * dim)
        """
        if not self._trained:
            raise  NotImplementedError("SOM object should call fit() before predict()")
        
        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'
        assert X.shape[1] == self.dim, f'This SOM has dimension {self.dim}. Rechieved input with dimension {X.shape[1]}'
        
        labels = [self.index_bmu(x) for x in X]
        labels = [(self.m*i + j) for i, j in labels]
        return labels
    
    def fit_predict(self, X : np.ndarray, epoch : int, shuffle=True):
        """Fit the model based on the data and return the prediciton of  the data

        Args:
            X (np.ndarray): Input data
            epoch (int): number of training iteration 
            shuffle (bool, optional): the initate data to be evaluate in the matrix. 
                Defaults to True.
        
        Returns:
            np.array(): the prediciton of each data
            
        Overall Time Complexity: O(epoch * N * m * n * dim)
        """
        self.fit(X = X, epoch = epoch, shuffle=shuffle) # O(epoch * N * m * n * dim)
        return self.predict() # O(N * m * n * dim)
    
    @property
    def cluster_center_(self):
        """Generate the list of all neurons

        Returns:
            np.ndarray(): list of all neurons with shape (m*n, dim)
        """
        return np.reshape(self.neurons, (-1, self.dim))
