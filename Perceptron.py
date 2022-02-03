from random import randint

# This file is responsible for
class Perceptron:
    def __init__(self, data, labels, classes=None):
        """
          A Perceptron is a binary classifier that learns how to split data
          into 2 given classes.

          Args:
            data: data the perceptron is going to train on
            labels: class labels for the data
            classes: what the classes stand for that isn't just 0 or 1
        """
        # Initiate data
        self.data = data
        self.labels = labels
        # How to interpret the labels
        self.classes = classes
        # Initialize random weights including the bias
        self.w = list([randint(-5, 5) for _ in range(len(data[0]) + 1)])

    def train_step(self, stop_acc=1):
        """
          Runs one training step of the model

          Args:
            stop_acc: Minimum accuracy to stop training at
        """
        # Get predictions on data
        predictions = self.predict(self.data)
        # Gather all the predictions that don't match the actual
        mislabled = []
        for i in range(len(predictions)):
            if predictions[i] != self.labels[i]:
                # If they don't match, add the data, prediction, and real label to our mislabled list
                # Add bias field to data as this is needed when updating weights later
                mislabled.append({
                    'data': [1] + self.data[i], 
                    'prediction': predictions[i], 
                    'label': self.labels[i]
                    })
        # Compute accuracy
        acc = 1-(len(mislabled)/len(predictions))
        # If model is accurate enough, no need to update weights, return accuracy
        if stop_acc <= acc:
            return acc
        # Update Weights by a random mislabled point
        mislabled_point = randint(0, len(mislabled)-1)
        if mislabled[mislabled_point]['prediction'] == 0:
            # Prediction says it is class 0, when we want it to be class 1, meaning x dot w is too small and must be increased
            self.w = list([self.w[i] + mislabled[mislabled_point]['data'][i] for i in range(len(self.w))])
        else:
            # Prediction says it is class 1, we want it to be class 0, meaning x dot w is too big and must be decreased
            self.w = list([self.w[i] - mislabled[mislabled_point]['data'][i] for i in range(len(self.w))])
        # Return current model accuracy
        return acc

    def train(self, steps, stop_acc=1):
        """
          Trains the model for a certain amount of steps or until it 
          reaches a certain level of accuracy

          Args:
            steps: the amount of training steps the model should take
            stop_acc: Minimum accuracy to stop training at
        """
        for _ in range(steps):
            acc = self.train_step(stop_acc)
            print(acc)
            if stop_acc <= acc:
                return acc
        return acc

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