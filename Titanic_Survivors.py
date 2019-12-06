import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("train.csv")                                      #Reading the training data csv file

# Cleaning of the Data:

columns_to_drop = ['PassengerId','Name','Ticket','Cabin','Embarked'] #Unnecessary columns which don't influence our prediction for any passenger

data_clean = data.drop(columns_to_drop, axis = 1)                    #Cleaning the data by removing the above unnecessary columns

le = LabelEncoder()
data_clean['Sex'] = le.fit_transform(data_clean['Sex'])              #Converting data into total numeric data (males-1, females-0)


data_clean = data_clean.fillna(data_clean['Age'].mean())

input_cols = ["Pclass","Sex","Age","SibSp","Parch","Fare"]
output_cols = ["Survived"]

X = data_clean[input_cols]
Y = data_clean[output_cols]


# function containing the definition of Entropy:
def entropy(col):
    
    counts = np.unique(col,return_counts=True)
    N = float(col.shape[0])
    
    ent = 0.0
        
    for ix in counts[1]:
        p = ix/N
        ent += (-1*p*np.log2(p))
    
    return ent


def divide_data(x_data, fkey, fval):
    
    # Work with Pandas Data Frames
    # Empty data frames created 
    x_right = pd.DataFrame([], columns=x_data.columns)
    x_left = pd.DataFrame([], columns=x_data.columns)
    
    # Will copy the data now
    for ix in range(x_data.shape[0]):
        val = x_data[fkey].loc[ix]
        
        # checking the value with the threshold value (fval)
        if val > fval:
            x_right = x_right.append(x_data.loc[ix])
        else:    
            x_left = x_left.append(x_data.loc[ix])
    
    return x_left, x_right


# function containing the definition of Information Gain:
def information_gain(x_data, fkey, fval):
    
    left, right = divide_data(x_data, fkey, fval)
    
    # % age of samples are on left and right
    l = float(left.shape[0]/x_data.shape[0])
    r = float(right.shape[0]/x_data.shape[0])
    
    # All examples come to one side!
    if left.shape[0] == 0 or right.shape[0] == 0:
        return -1000000 # Min Informtion Gain
    
    i_gain = entropy(x_data.Survived) - (l*entropy(left.Survived) + r*entropy(right.Survived))
    
    return i_gain

for fx in X.columns:
    print(fx)
    print(information_gain(data_clean, fx, data_clean[fx].mean()))


# class of Decision Tree:
class DecisionTree:
    
    # Constructor
    def __init__(self, depth=0, max_depth=5):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None # conclusion for the node
        
    def train(self, X_train):
        
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        info_gains = []
        
        for ix in features:
            i_gain = information_gain(X_train, ix, X_train[ix].mean())
            info_gains.append(i_gain)
        
        self.fkey = features[np.argmax(info_gains)]
        self.fval = X_train[self.fkey].mean()
        print("Making Tree Features is",self.fkey)
        
        # Split Data
        data_left, data_right = divide_data(X_train, self.fkey, self.fval)
        data_left = data_left.reset_index(drop=True)
        data_right = data_right.reset_index(drop=True)
        
        # Truly a left node (Base Case 1)
        if data_left.shape[0] == 0 or data_right.shape[0] == 0:
            if X_train.Survived.mean() >= 0.5:
                self.target = "Survive"
            else:
                self.target = "Dead"
            return
        
        # Stop early when depth >= max_depth (Base Case 2)
        if self.depth >= self.max_depth:
            if X_train.Survived.mean() >= 0.5:
                self.target = "Survive"
            else:
                self.target = "Dead"
            return
        
        # Recursive Case
        self.left = DecisionTree(depth = self.depth+1, max_depth=self.max_depth)
        self.left.train(data_left)
        
        self.right = DecisionTree(depth = self.depth+1, max_depth=self.max_depth)
        self.right.train(data_right)
        
        # We can set the target at every node
        if X_train.Survived.mean() >= 0.5:
            self.target = "Survive"
        else:
            self.target = "Dead"
            return
    
    def predict(self, test):
        
        if test[self.fkey] > self.fval:
            # go to right
            if self.right is None:
                return self.target
            return self.right.predict(test)
        else:
            if self.left is None:
                return self.target
            return self.left.predict(test)


dt = DecisionTree()

dt.train(data_clean)

data_test = pd.read_csv("test.csv")                                  #Reading the training data csv file

columns_to_drop = ['PassengerId','Name','Ticket','Cabin','Embarked'] #Unnecessary columns which don't influence our prediction for any passenger

data_test_clean = data_test.drop(columns_to_drop, axis = 1)          #Cleaning the data by removing the above unnecessary columns

le = LabelEncoder()
data_test_clean['Sex'] = le.fit_transform(data_test_clean['Sex'])    #Converting data into total numeric data (males-1, females-0)

y_pred = []
for ix in range(data_test_clean.shape[0]):
    y_pred.append(dt.predict(data_test_clean.loc[ix]))

le = LabelEncoder()
y_pred = le.fit_transform(y_pred)

# creating the final output (.csv) file
with open("titanic_submission.csv", 'w', encoding = "UTF-8") as f:
    f.write('Passenger')
    f.write(',')
    f.write('Survived')
    f.write('\n')
    for ix in range(y_pred.shape[0]):
        id = str(892+ix)
        f.write(id)
        f.write(',')
        f.write(str(y_pred[ix]))
        f.write('\n')