from random import randint

# This file is responsible for
class Perceptron:
    def __init__(self, data, labels, classes=None, max_steps=200):
        # Initiate data with bias attached to first field
        self.data = [[1] + row for row in data]
        self.labels = labels
        # How to interpret the labels
        self.classes = classes
        # Max amount of steps this can train for before stopping
        self.max_steps = max_steps
        # Initialize random weights including the bias
        self.w = list([randint(-5, 5) for _ in range(len(data[0]) + 1)])

    def train_step(self):
        """
          Runs one training step of the model
        """
        # Get predictions on data
        predictions = self.predict(self.data)
        # Gather all the predictions that don't match the actual
        mislabled = []
        for i in range(len(predictions)):
            if predictions[i] != self.labels[i]:
                # If they don't match, add the data, prediction, and real label to our mislabled list
                mislabled.append({
                    'data': self.data[i], 
                    'prediction': predictions[i], 
                    'label': self.labels[i]
                    })
        # Update Weights by a random mislabled point
        mislabled_point = randint(0, len(mislabled)-1)
        if mislabled[mislabled_point]['prediction'] == 0:
            # Prediction says it is class 0, when we want it to be class 1, meaning x dot w is too small and must be increased
            self.w = list([self.w[i] + mislabled[mislabled_point]['data'][i] for i in range(len(self.w))])
        else:
            # Prediction says it is class 1, we want it to be class 0, meaning x dot w is too big and must be decreased
            self.w = list([self.w[i] - mislabled[mislabled_point]['data'][i] for i in range(len(self.w))])
        # Return current model accuracy
        return 1-(len(mislabled)/len(predictions))

    def predict(self, data):
        """
          Predicts the class number the data belongs to

          Args:
            data: the data to make predictions on
        """
        # Checks if the dot product of the data is positive (class 1)
        # or negative (class 0)
        return [int(dot(x, self.w) >= 0) for x in data]

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