import pandas as pd
import numpy as np
from itertools import cycle
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from empiricaldist import Cdf
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree


def generate_fake_dataframe(size, cols, col_names=None, intervals=None, seed=None):
    categories_dict = {

        'animals': ['cow', 'rabbit', 'duck', 'shrimp', 'pig', 'goat', 'crab', 'deer', 'bee', 'sheep', 'fish', 'turkey',
                    'dove', 'chicken', 'horse'],
        'names': ['James', 'Mary', 'Robert', 'Patricia', 'John', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth',
                  'Ahmed', 'Barbara', 'Richard', 'Susan', 'Salomon', 'Juan Luis'],
        'cities': ['Stockholm', 'Denver', 'Moscow', 'Marseille', 'Palermo', 'Tokyo', 'Lisbon', 'Oslo', 'Nairobi',
                   'Río de Janeiro', 'Berlin', 'Bogotá', 'Manila', 'Madrid', 'Milwaukee'],
        'colors': ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'purple', 'pink', 'silver', 'gold', 'beige',
                   'brown', 'grey', 'black', 'white']
        }
    default_intervals = {"i": (0, 10), "f": (0, 100), "c": ("names", 5),
                         "d": ("2020-01-01", "2020-12-31")}
    rng = np.random.default_rng(seed)

    first_c = default_intervals["c"][0]
    categories_names = cycle([first_c] + [c for c in categories_dict.keys() if c != first_c])
    default_intervals["c"] = (categories_names, default_intervals["c"][1])

    if isinstance(col_names, list):
        assert len(col_names) == len(
            cols), f"The fake DataFrame should have {len(cols)} columns but col_names is a list with {len(col_names)} elements"
    elif col_names is None:
        suffix = {"c": "cat", "i": "int", "f": "float", "d": "date"}
        col_names = [f"column_{str(i)}_{suffix.get(col)}" for i, col in enumerate(cols)]

    if isinstance(intervals, list):
        assert len(intervals) == len(
            cols), f"The fake DataFrame should have {len(cols)} columns but intervals is a list with {len(intervals)} elements"
    else:
        if isinstance(intervals, dict):
            assert len(
                set(intervals.keys()) - set(default_intervals.keys())) == 0, f"The intervals parameter has invalid keys"
            default_intervals.update(intervals)
        intervals = [default_intervals[col] for col in cols]
    df = pd.DataFrame()
    for col, col_name, interval in zip(cols, col_names, intervals):
        if interval is None:
            interval = default_intervals[col]
        assert (len(interval) == 2 and isinstance(interval, tuple)) or isinstance(interval,
                                                                                  list), f"This interval {interval} is neither a tuple of two elements nor a list of strings."
        if col in ("i", "f", "d"):
            start, end = interval
        if col == "i":
            df[col_name] = rng.integers(start, end, size)
        elif col == "f":
            df[col_name] = rng.uniform(start, end, size)
        elif col == "c":
            if isinstance(interval, list):
                categories = np.array(interval)
            else:
                cat_family, length = interval
                if isinstance(cat_family, cycle):
                    cat_family = next(cat_family)
                assert cat_family in categories_dict.keys(), f"There are no samples for category '{cat_family}'. Consider passing a list of samples or use one of the available categories: {categories_dict.keys()}"
                categories = rng.choice(categories_dict[cat_family], length, replace=False, shuffle=True)
            df[col_name] = rng.choice(categories, size, shuffle=True)
        elif col == "d":
            df[col_name] = rng.choice(pd.date_range(start, end), size)
    return df

def correlation(dataframe):
    print("correlation...")

    dataframe = dataframe.apply(pd.to_numeric) # convert all columns of DataFrame

    top = cm.get_cmap('Oranges_r', 128)
    bottom = cm.get_cmap('Blues', 128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')
    corr = dataframe.corr()
    # Create positive correlation matrix
    corr = dataframe.corr().abs()
    # Create and apply mask
    mask = np.triu(np.ones_like(corr, dtype=bool))
    tri_df = corr.mask(mask)
    # Find columns that meet treshold
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.9)]
    print(to_drop)
    reduced_df = dataframe.drop(to_drop,axis=1)
    print("Dimensionality reduced from {} to {}.".format(dataframe.shape[1], reduced_df.shape[1]))
    #Insert Column without erro

    # Create and apply mask
    mask = np.triu(np.ones_like(reduced_df.corr(), dtype=bool))
    sns.heatmap(reduced_df.corr(), mask=mask,
                center=0, cmap=newcmp, linewidths=1,
                annot=True, fmt=".2f")
    plt.tight_layout()

    plt.show()
    return  reduced_df;
if __name__ == '__main__':
    df1 = generate_fake_dataframe(
        size=100,
        cols="cccccccci",
        col_names=["Pergunta_1","Pergunta_2","Pergunta_3","Pergunta_4"
                   ,"Pergunta_5","Pergunta_6","Pergunta_7","Pergunta_8","Evasao"],
        intervals={"c":['1','4','7'],"i":[0,2]},
        seed=42)
    correlation(df1)
    target = df1['Evasao']
    X = df1.copy().drop(columns=['Evasao']);
    # Fit the classifier with default hyper-parameters
    clf = DecisionTreeClassifier(criterion="entropy", max_depth = 5)
    model = clf.fit(X, target)

    text_representation = tree.export_text(clf)
    print(text_representation)

    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, target,
                                                                    test_size=0.3, random_state=3)
    featureNames = df1.columns[0:8]
    print(featureNames)
    fig = plt.figure(figsize=(13,13))
    targetNames = df1["Evasao"].unique().tolist()
    tree.plot_tree(clf,feature_names = featureNames,
               class_names=['Nao Evasão','Evasão'],
               filled = True)

    plt.show()


