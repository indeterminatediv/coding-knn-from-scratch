import numpy as np
from collections import Counter
class MeraKNN:
    
    def __init__(self,K=5):
        self.X_train = None
        self.Y_train = None
        self.n_neighbors = K
    
    def fit(self,X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        y_pred = []
        # calculating distances from each training point
        for i in X_test:
            distances = []
            for j in self.X_train:
                distances.append(self.calculate_distance(i,j))
            n_neighbors = sorted(list(enumerate(distances)),key=lambda x:x[1])[0:self.n_neighbors]
            label = self.majority_count(n_neighbors)    
            y_pred.append(label)
        return np.array(y_pred)    
        
    def calculate_distance(self, point_A, point_B):
        return np.linalg.norm(point_A-point_B)    
        
    def majority_count(self,neighbors):
        votes = []
        for i in neighbors:
            votes.append(self.y_train[i[0]])
        votes = Counter(votes)    
        return votes.most_common()[0][0]