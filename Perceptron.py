from random import randint

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
        self.classes = classes
        # Store metrics that evaluate model
        self.metrics = metrics
        # Initialize weight variable
        self.w = None

    def train_step(self, data, labels):
        """
          Runs one training step of the model.

          Args:
            data: the data to train on
            labels: the labels for the data
        """
        if not self.w:
            self.w = list([randint(-5, 5) for _ in range(len(data[0]) + 1)])
        # Get predictions on data
        predictions = self.predict(data)
        # Gather all the predictions that don't match the actual
        mislabled = []
        for i in range(len(predictions)):
            if predictions[i] != labels[i]:
                # If they don't match, add the data, prediction, and real label to our mislabled list
                # Add bias field to data as this is needed when updating weights later
                mislabled.append({
                    'data': [1] + data[i], 
                    'prediction': predictions[i], 
                    'label': labels[i]
                    })
        # Update Weights by a random mislabled point
        mislabled_point = randint(0, len(mislabled)-1)
        if mislabled[mislabled_point]['prediction'] == 0:
            # Prediction says it is class 0, when we want it to be class 1, meaning x dot w is too small and must be increased
            self.w = list([self.w[i] + mislabled[mislabled_point]['data'][i] for i in range(len(self.w))])
        else:
            # Prediction says it is class 1, we want it to be class 0, meaning x dot w is too big and must be decreased
            self.w = list([self.w[i] - mislabled[mislabled_point]['data'][i] for i in range(len(self.w))])
        # Return how the model is doing based on passed in parameters
        return {key: self.metrics[key](list(self.predict(data)), labels) for key in self.metrics.keys()}

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
            for key in stop_metrics.keys():
                if results[key] < stop_metrics[key]:
                    # If any key is less then the stop metric, keep training
                    continue
                else:
                    # Every key is greater than or equal to the stop metric, training is done
                    return results
        # Return final results
        return results

    def predict(self, data):
        """
          Predicts the class number the data belongs to

          Args:
            data: the data to make predictions on
        """
        # Checks if the dot product of the data is positive (class 1)
        # or negative (class 0). Adds bias vector at the top
        return [int(dot([1] + x, self.w) >= 0) for x in data]

    def predict_classes(self, data):
        """
          Predicts the classes of the inputted data rather than just the number

          Args:
            data: the data to make predictions on
        """
        if self.classes:
            # Translates class 0 or class 1 to the real labels if provided
            return [self.classes[pred] for pred in self.predict(data)]
        return self.predict(data)
        

def dot(a, b):
    """
      Computes the dot product of a and b

      Args:
        a: A vector
        b: A vector
    """
    if len(a) != len(b):
        raise Exception(f'Vector {a} and Vector {b} are not the same size.')
    return sum([a[i] * b[i] for i in range(len(a))])

def accuracy(pred, actual):
    """
      Calculates the accuracy of a classifier

      Args:
        pred: Predicted class labels
        actual: The actual class labels
    """
    return sum([int(pred[i] == actual[i]) for i in range(len(actual))])/len(actual)