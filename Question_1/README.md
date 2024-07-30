# Random Forest Classifier with MNIST dataset
## 1. Random Forest
A Random Forest is an ensemble learning algorithm used for classification and regression tasks. It operates by constructing a multitude of decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

## 2. Structure of Random Forest
Random Forest includes:
- $N$ Decision Trees built by Bootstrapped Dataset using Predefined Conditions.
- From $N$ trees, we get $N$ predictions.
- The $N$ predictions are aggregated through an Ensemble Bagging method to produce the final result.

## 3. How it works
- **Step 1**: From the initial data, create a Bootstrapped Dataset based on it.
Bootstrapped Dataset is a dataset randomly selected from the original dataset. The process works by assigning each sample an equal probability, then randomly selecting data until the generated data has the same size as the original dataset (duplicates are allowed).
- **Step 2**: Create Decision Trees using the Predefined Conditions method on the Bootstrapped dataset. The method is as follows:
- **Step 2-1**: Randomly select 2 features from the dataset.
- **Step 2-2**: Build a Decision Node based on comparing those 2 features; the chosen feature is removed from the dataset.
- **Step 2-3**: Repeat **Step 2-1 → 2-2** to find the optimal Decision Tree.
- **Step 3**: Repeat Steps 1 → 2 to find $N$ Decision Trees.
- **Step 4**: When new data is introduced for prediction, the $N$ Decision Trees will return $N$ results. Using a simple Ensemble method, Majority Voting, the final result is produced.


*See more detail explain Random Forest model in: [Bài 5 - Random Forest [Thịnh Diablog]](https://flowery-fairy-f0d.notion.site/B-i-5-Random-Forest-d39ed94c6c1240c0b87f1708e5358f12?pvs=4)*

# 4. Files
- `main.py`: Simple pipeline using Random Forest Classifier in MNIST dataset
- `model.py`: Random Forest implementation from scratch using Numpy
- `utils.py`: Load MNIST dataset
- `notebook/Random_Forest_with_MNIST_Dataset.ipynb`: Example of implementation for Question 1