import numpy as np

def get_random_centroids(X, k):

    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    
    centroids = []
    np.random.seed(32)

    for i in range(k):
        centroids.append(np.random.randint(0,150,3))
  
   
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float) 



def lp_distance(X, centroids, p=2):

    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    for i in range(k):
        
        distances.append(np.power(np.sum(np.power(np.abs(X - centroids[i]),p),axis=1),(1/p)))
    
    return np.asarray(distances).T

def assign(X,d):
    classes = np.argmin(d,axis = 1).reshape(X.shape[0],1)
    #print(f"X shape is{X.shape}, and clases shape is {classes.shape}")
    assignment = np.hstack((X,classes))
    return assignment

def centroids_calc(Y,k):
    centroids = np.empty((k,3)) 
    for i in range(k):
            pixels_per_k = Y[Y[:,-1] == i][:,0:-1]
            new_centroid = np.mean(pixels_per_k, axis = 0)
            #new_centroid = new_centroid.reshape(3,1)
            #print(new_centroid.shape, type(new_centroid))
            centroids[i] = new_centroid
    
    return centroids
    
    
def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    classes = np.empty(X.shape[0])

    for i in range(max_iter):
        distances = lp_distance(X,centroids,p)
        Y = assign(X, distances)
        new_centroids = centroids_calc(Y,k)
        if np.all(new_centroids == centroids):
            break
        else:
            centroids = centroids_calc(Y,k)
    classes = Y[:,-1]
    
    return centroids, classes

def distribution_calc(X,centroids, i):
    distances = lp_distance(X,centroids[0:i+1],2)
    distances = np.power(np.min(distances,axis = 1),2)
    probability =  distances / np.sum(distances)
    return probability

def get_weighted_centroids(X,k):
    centroids = np.empty((k,3))
    for i in range(k):
        if i == 0:
            random_row = np.random.choice(X.shape[0])
            centroids[i] = X[random_row]
        else:
            prob = distribution_calc(X,centroids, i)
            random_row = np.random.choice(X.shape[0], p = prob)
            centroids[i] = X[random_row]
    
    return centroids
    
    
    

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_weighted_centroids(X, k)
    classes = np.empty(X.shape[0])

    for i in range(max_iter):
        distances = lp_distance(X,centroids,p)
        Y = assign(X, distances)
        centroids = centroids_calc(Y,k)
    classes = Y[:,-1]
    
    return centroids, classes

