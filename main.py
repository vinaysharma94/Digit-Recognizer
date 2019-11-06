import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random 

data = pd.read_csv("data/train.csv").as_matrix()

clf = DecisionTreeClassifier()

train_inputs = data[0:21000, 1:]
train_outputs = data[0:21000, 0]
test_inputs = data[21000:, 1:]
test_outputs = data[21000:, 0]

# train the classifier
clf.fit(train_inputs, train_outputs)

# test
print("\tPredicted Result\t|\tCorrect Result\t")
nb_tests = 9
correct_predictions_count = 0.0
for i in range(0, nb_tests):
    # take random set from the dataset
    test_index = random.randint(0, len(test_inputs) - 1)
    predicted_result = clf.predict([test_inputs[test_index]])
    correct_result = test_outputs[test_index]
    
    if predicted_result == correct_result:
        correct_predictions_count += 1.0

    print("\t\t%d\t\t|\t\t%d\t" % (predicted_result, correct_result))

    d = test_inputs[test_index]
    d.shape = (28, 28)
    pt.imshow(255-d, cmap='gray')
    pt.show()
print("Accuracy: %.2f%%" % (correct_predictions_count / nb_tests * 100.0))
