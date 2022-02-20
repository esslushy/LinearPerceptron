import numpy as np
import random

class Perceptron:
    def __init__(self, classes=None, metrics={}):
        """
          A Perceptron is a binary classifier that learns how to split data
          into 2 given classes.

          Args:
            classes: what the classes stand for that isn't just 0 or 1
            metrics: what metrics to judge
        """
        # How to interpret the labels
        if classes and len(classes) != 2:
            raise Exception('Classes must be an array of length 2')
        self.classes = classes
        # Store metrics that evaluate model
        self.metrics = metrics
        # Initialize weight variable
        self.w = np.array([])

    def train_step(self, data, labels):
        """
          Runs one training step of the model.

          Args:
            data: the data to train on
            labels: the labels for the data
        """
        # Check if we need to initialize
        if self.w.size == 0:
            # One more to account for bias point
            self.w = np.random.random(len(data[0]) + 1)
        # Get predictions on data
        predictions = self.predict(data)
        # Find all points with mislabeled
        mislabled = (predictions != labels)
        # Get the mislabled data points
        mislabled_data = data[mislabled]
        mislabled_label = labels[mislabled]
        # Select a random data point
        data_point = random.randint(0, len(mislabled_data) - 1)
        # Update on that data point
        if mislabled_label[data_point] == 1:
            # Prediction says it is class 0, when we want it to be class 1, meaning x dot w is too small and must be increased
            self.w += np.insert(mislabled_data[data_point], 0, 1)
        else:
            # Prediction says it is class 1, we want it to be class 0, meaning x dot w is too big and must be decreased
           self.w -= np.insert(mislabled_data[data_point], 0, 1)
        # Return how the model is doing based on passed in parameters
        new_predictions = self.predict(data)
        return {key: self.metrics[key](new_predictions, labels) for key in self.metrics.keys()}

    def train(self, data, labels, steps=200, stop_metrics={}):
        """
          Trains the model for a certain amount of steps or until it 
          passes all stop metric requirements

          Args:
            data: the data to train on
            labels: the labels for the data
            steps: the amount of training steps the model should take
            stop_metrics: the metrics the model will watch and stop if it reaches the minimum value in all of them 
        """
        for _ in range(steps):
            # Run the training
            results = self.train_step(data, labels)
            # Check if it passes all stop metrics
            if all([results[key] >= stop_metrics[key] for key in stop_metrics.keys()]):
                return results
        # Return final results
        return results

    def predict(self, data):
        """
          Predicts the class number the data belongs to

          Args:
            data: the data to make predictions on
        """
        # Add bias to data (insert one at the top of each data point)
        data = np.insert(data, 0, 1, axis=1)
        # Computes the dot product of each piece of data with the perceptron weights
        predictions = np.dot(data, self.w)
        # Turn the raw predictions into 0 or 1 class predictions
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = 0
        return predictions

    def predict_classes(self, data):
        """
          Predicts the classes of the inputted data rather than just the number

          Args:
            data: the data to make predictions on
        """
        predictions = self.predict(data)
        if self.classes:
            # Translates class 0 or class 1 to the real labels if provided
            predictions[predictions == 0] = self.classes[0]
            predictions[predictions == 1] = self.classes[1]
        return predictions

def accuracy(pred, actual):
    """
      Calculates the accuracy of a classifier

      Args:
        pred: Predicted class labels
        actual: The actual class labels
    """
    return sum(pred == actual)/len(pred)