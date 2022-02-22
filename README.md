# Linear Perceptron
Recently, in my Mathematics of Data Models course, we learned how a linear perceptron worked. I found it very interesting and decided to write my own linear perceptron in python. This was not assigned for any class and was completely self led. I utilized Seaborn's penguin dataset for this project. 

# To Run
1. Run `pip install -r requirements.txt`
2. Run `python Main.py`

# Note
I showed this repository to my professor, and he had a few recommendations that I decided to implement. They are:
- Use Numpy arrays and dot products
    - Understand view vs copy in numpy as well
- Boolean indexing using numpy when judging across all data points
- Boolean indexing for faster label checking
- Batch learning (Run against everything all at once and update it based on all teh information)
- Test how my model compares against scikit learn