# Project dt047a

## Methods

### Descision Tree

How can we know based on input whether its white or red wine?

Dataset
● Red wine (1599 instances)
● White wine ( 4898 instances)

Data mining method
● Decision tree (Predict yes or no outcome of varying variables/attributes)

Training and Testing
● Bootstrap datasets and use holdout for dividing training and test sets

Output
● Whether its white or red wine based on attributes

Evaluation Methods
● leave-one-out cross validation (Prediction accuracy for decision tree algorithm)
● Confusion matrix
○ Whether the predicted value of white/red wine actually is white/red wine
○ Matthews correlation matrix to evaluate the confusion matrix, determine the accuracy.

### Multiple Linear Regression

How could we know the quality of newly produced wines?

Dataset
● White wine ( 4898 instances)

Data mining method
● Multiple Linear Regression
○ Serves multiple variables to predict outcome of different variables

Training and Testing
● Bootstrap datasets and use holdout for dividing training and test sets

Output
● Wine quality learned attributes

Evaluation Method
● Backward elimination (eliminates all non-significant variables)
