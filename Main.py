import seaborn as sns
from Perceptron import Perceptron, accuracy
from sklearn.model_selection import KFold
from sklearn import linear_model

# Load dataframe
sns.set(font_scale=1.5)
df = sns.load_dataset('penguins')

# We are going to try and find the difference between
# Adelie and Gentoo penguins using all fields
relevant_classes = ('Adelie', 'Gentoo')

# Remove all inputs that are NaN
df.dropna(inplace=True)

# Pull out only relevant classes. The | is bitwise or which compares each list and makes
# a new list where a[x] | b[x] = c[x]
df = df[(df.species == relevant_classes[0]) | (df.species == relevant_classes[1])]

# Pull out labels
labels = df['species']
# Drop unused data (island, body mass, and sex) and species (redundant)
df = df.drop(['island', 'body_mass_g', 'sex', 'species'], axis=1)

# Z-score normalization. Tells how far each data point is off the mean.
for col in df.columns:
    mean = df[col].mean()
    std_dev = df[col].std()
    df = df.apply(lambda x: (x - mean) / std_dev)

# Turn our labels into class labels (0 or 1)
labels = labels.apply(lambda x: 0 if x == relevant_classes[0] else 1)

# Let's take a look at our ready data
print(labels, df)

# Turn it into lists for our model to consume
x = df.values
y = labels.values

# Let's train our model with k-fold cross validation to ensure that it is reliable

# Fold across all our data in 80-20 split train to test changing whcih data is 
# training and testing each time to ensure our model works well.
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(x):
    # Create new perceptron
    penguin_perceptron = Perceptron(relevant_classes, {'accuracy': accuracy})
    # Train it
    train_results = penguin_perceptron.train(x[train_index], y[train_index], 500, {'accuracy': 0.95})
    print('Training results:')
    print(train_results)
    # Test it and get results
    test_accuracy = accuracy(penguin_perceptron.predict(x[test_index]), y[test_index])
    print('Testing accuracy: ')
    print(test_accuracy)
    print('Sklearn model for comparison')
    sklearn_perceptron = linear_model.Perceptron()
    sklearn_perceptron.fit(x[train_index], y[train_index])
    print(sklearn_perceptron.score(x[test_index], y[test_index]))