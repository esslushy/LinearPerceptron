import seaborn as sns
from Perceptron import Perceptron, accuracy

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

# Create the perceptron
penguin_perceptron = Perceptron(df.values.tolist(), labels.values.tolist(), relevant_classes, {'accuracy': accuracy})

# Train it for 100 steps printing out the accuracy
print(penguin_perceptron.train(100, {'accuracy': 0.95}))
print(penguin_perceptron.predict_classes(df.values.tolist()))