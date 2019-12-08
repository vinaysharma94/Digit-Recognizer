######################################################
"""Hand Digit Recognition using MNIST dataset.
Classifier Used: KNN, Decision Tree, Random Forest and Linear SVC."""
######################################################

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import random 

data = pd.read_csv("data/train.csv").as_matrix()

clf1 = KNeighborsClassifier() #KNN Classifier
clf2 = DecisionTreeClassifier() # Decision Tree Classifier
clf3 = RandomForestClassifier() # Random Forest Classifier
clf4 = LinearSVC() # LinearSVC

train_inputs = data[0:21000, 1:]
train_outputs = data[0:21000, 0]
test_inputs = data[21000:, 1:]
test_outputs = data[21000:, 0]

while True:
	print("\n1. KNeighbours (KNN)\n2. Decision Tree Classifier\n2. Random Forest Classifier\n4. Linear SVC")
	
	n = input('Enter Classifier number: ')
	if n == "1":
		###### KNeighbors(KNN) Classifier ########
		# Train the KNeighbors(KNN) Classifier
		clf1.fit(train_inputs, train_outputs)

		# Test of KNeighbors(KNN) Classifier
		print("\n\n******************* KNEIGHBORS(KNN) CLASSIFIER *****************************\n")

		print("\tPredicted Result\t|\tCorrect Result\t")
		nb_tests = 9
		correct_predictions_count = 0.0
		for i in range(0, nb_tests):
			# Take random set from the dataset
			test_index = random.randint(0, len(test_inputs) - 1)
			predicted_result = clf1.predict([test_inputs[test_index]])
			correct_result = test_outputs[test_index]
    
			if predicted_result == correct_result:
				correct_predictions_count += 1.0

			print("\t\t%d\t\t|\t\t%d\t" % (predicted_result, correct_result))

			d = test_inputs[test_index]
			d.shape = (28, 28)
			pt.imshow(255-d, cmap='gray')
			pt.show()
		print("Accuracy with KNeighbors Classifier: %.2f%%" % (correct_predictions_count / nb_tests * 100.0))
		print("\n\n")

###### End of KNeighbors(KNN) Classifier ########

	elif n == "2":
		#Train Decision Tree Classifier
		clf2.fit(train_inputs, train_outputs)
		print("******************* DECISION TREE CLASSIFIER *****************************\n")

		print("\tPredicted Result Decition Tree \t|\tCorrect Result\t")
		nb_tests = 9
		correct_predictions_count = 0.0
		for i in range(0, nb_tests):
			# take random set from the dataset
			test_index = random.randint(0, len(test_inputs) - 1)
			predicted_result = clf2.predict([test_inputs[test_index]])
			correct_result = test_outputs[test_index]
    
			if predicted_result == correct_result:
				correct_predictions_count += 1.0

			print("\t\t%d\t\t|\t\t%d\t" % (predicted_result, correct_result))

			d = test_inputs[test_index]
			d.shape = (28, 28)
			pt.imshow(255-d, cmap='gray')
			pt.show()
		print("Accuracy with Decision Tree Classifier:  %.2f%%" % (correct_predictions_count / nb_tests * 100.0))
		print("\n\n")
##### End of Decision Tree Classifier ###########

	elif n == "3":
		##### Random Forest Classifier ######

		#Train Random Forest Classsifier
		clf3.fit(train_inputs, train_outputs)

		print("******************* RANDOM FOREST CLASSIFIER *****************************\n")
		print("\tPredicted Result \t|\tCorrect Result\t")
		nb_tests = 9
		correct_predictions_count = 0.0
		for i in range(0, nb_tests):
			# take random set from the dataset
			test_index = random.randint(0, len(test_inputs) - 1)
			predicted_result = clf3.predict([test_inputs[test_index]])
			correct_result = test_outputs[test_index]
    
			if predicted_result == correct_result:
				correct_predictions_count += 1.0

			print("\t\t%d\t\t|\t\t%d\t" % (predicted_result, correct_result))

			d = test_inputs[test_index]
			d.shape = (28, 28)
			pt.imshow(255-d, cmap='gray')
			pt.show()
		print("Accuracy with Random Forest Classifier:  %.2f%%" % (correct_predictions_count / nb_tests * 100.0))

##### End of Random Forest Classifier #####
	elif n == "4":
		##### SVM ######

		#Train SVM Classsifier
		clf4.fit(train_inputs, train_outputs)

		print("******************* LINEAR SVC*****************************\n")
		print("\tPredicted Result \t|\tCorrect Result\t")
		nb_tests = 9
		correct_predictions_count = 0.0
		for i in range(0, nb_tests):
			# take random set from the dataset
			test_index = random.randint(0, len(test_inputs) - 1)
			predicted_result = clf4.predict([test_inputs[test_index]])
			correct_result = test_outputs[test_index]
    
			if predicted_result == correct_result:
				correct_predictions_count += 1.0

			print("\t\t%d\t\t|\t\t%d\t" % (predicted_result, correct_result))

			d = test_inputs[test_index]
			d.shape = (28, 28)
			pt.imshow(255-d, cmap='gray')
			pt.show()
		print("Accuracy with SVM Classifier:  %.2f%%" % (correct_predictions_count / nb_tests * 100.0))
##### End of SVM Classifier #####
	else:
		print("Invalid Input!!! Try Again")
