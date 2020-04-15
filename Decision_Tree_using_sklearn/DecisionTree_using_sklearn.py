import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
data = pd.read_csv("/home/thedarkcoder/Desktop/ML/Coursera/datasets/drug200.csv")
features = data[['Age','Sex','BP','Cholesterol','Na_to_K']].values
# print(features)

le = preprocessing.LabelEncoder()
le.fit(['M','F'])
features[:,1] = le.transform(features[:,1])

le = preprocessing.LabelEncoder()
le.fit(['HIGH','NORMAL','LOW'])
features[:,2] = le.transform(features[:,2])

le = preprocessing.LabelEncoder()
le.fit(['HIGH','NORMAL'])
features[:,3] = le.transform(features[:,3])
# print(features)

y = data['Drug']
x_train, x_test, y_train, y_test = train_test_split(features, y, test_size = 0.2, random_state = 3)
# print(type(x_test))

drugtree = DecisionTreeClassifier(criterion= 'entropy', max_depth= 4)
drugtree.fit(x_train, y_train)

pred_tree = drugtree.predict(x_test)
print(pred_tree[:10], y_test[:10])

print("Accuracy : ", metrics.accuracy_score(y_test, pred_tree))

