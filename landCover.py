import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from osgeo import gdal

# List of attributes which are used
attributes = ['Band 1', 'Band 2', 'Band 3', 'NDVI', 'NDWI']

# List of labels
labelId = {
    "Residential Land": 1,
    "Rice Paddies": 2,
    "Croplands": 3,
    "Grassland": 4,
    "Barrenland": 5,
    "Scrub": 6,
    "Forrest": 7,
    "Open Water": 8,
    "Aquaculture": 9,
}

# Read the samples file
train=pd.read_csv("./train.csv")
X = train.filter(attributes)
y = train.filter(['Label'])

# split samples file into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, stratify=y)

# --------------Random forest model---------------
param_grid = {
    'max_depth': range(8, 15, 1),
    'min_samples_split': range(3, 8, 1),
}
baseClf = RandomForestClassifier(random_state= 0)

clf = HalvingGridSearchCV(baseClf, param_grid, cv=5,
                          factor=2).fit(X_train, y_train)

print(clf.best_params_)

# --------------K-Neighbors model-----------------

# param_grid = {
#     'n_neighbors': range(4, 10, 1), 
#     'leaf_size': range(25, 40, 1), 
#     'p': [1, 2],
#     'weights': ['uniform', 'distance'],
#     'metric': ['minkowski', 'chebyshev']
# }

# baseClf = KNeighborsClassifier()

# clf = HalvingGridSearchCV(baseClf, param_grid, cv=5,
#                           factor=2).fit(X_train, y_train)

# print(clf.best_params_)

# ----------------------MLP model----------------------
# param_grid = {
#     'hidden_layer_sizes': [(50,50,50), (50,100,50), (25, 60, 60), (100,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.05],
#     'learning_rate': ['constant','adaptive'],
# }

# baseClf = MLPClassifier(random_state=0)

# clf = HalvingGridSearchCV(baseClf, param_grid, cv=5,
#                           factor=2).fit(X_train, y_train)

# print(clf.best_params_)

# -------------------SVC model-------------------- 
# param_grid = {
#     'C': [0.1, 1, 10, 100], 
#     'gamma': [1, 0.1, 0.01, 0.001],
#     'kernel': ['rbf', 'sigmoid'],
# }
# baseClf = SVC(random_state= 0)

# clf = HalvingGridSearchCV(baseClf, param_grid, cv=5,
#                           factor=2).fit(X_train, y_train)

# print(clf.best_params_)

#---------------------------------------------------

# predict the test set
y_pred=clf.predict(X_test)

# show the result of the predict on test set
print('Accuracy score - Test dataset: {}'.format(accuracy_score(y_test, y_pred)))
print('F1 score - Test dataset: {}'.format(f1_score(y_test, y_pred, average= 'macro')))
cm=confusion_matrix(y_test,y_pred)
f, ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_test")
plt.show()

# read the map
ds = gdal.Open(r'.\dataImage.tif', gdal.GA_ReadOnly)
data = ds.ReadAsArray()
bands, rows, cols = data.shape

rsl = []
for i in range(rows):
    for j in range(cols):
        if data[0, i, j] > -9999:
            tmp = np.append([i, j], data[:, i, j])
            rsl.append(tmp)

df = pd.DataFrame(rsl, columns=['rows', 'cols', 'Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5', 'Band 6', 'Band 7'])

df.to_csv('map.csv', index=False)

map_df = pd.read_csv(r'.\map.csv')

# calculate NDVI and NDWI for the map
map_df["NDVI"] = (map_df["Band 4"] - map_df["Band 1"])/(map_df["Band 4"] + map_df["Band 1"])
map_df["NDWI"] = (map_df["Band 2"] - map_df["Band 4"])/(map_df["Band 4"] + map_df["Band 2"])


tmp = map_df.filter(attributes)

# predict the map
map_df['pred'] = clf.predict(tmp)

# create land cover map file
map_df = map_df[['rows', 'cols', 'pred']]
ds = gdal.Open(r'.\dataImage.tif', gdal.GA_ReadOnly)
data = ds.ReadAsArray()
bands, rows, cols = data.shape

output_map = np.zeros(shape=(rows, cols), dtype=np.float32) + 255
for lines in map_df.values:
    i, j, pred = lines
    output_map[int(i)][int(j)] = labelId[pred]


outfname ='LCmap.output.tif'
driver = gdal.GetDriverByName("GTiff")
dst_ds = driver.Create(outfname, cols, rows, 1, gdal.GDT_Float32)
dst_ds.SetGeoTransform(ds.GetGeoTransform())
dst_ds.SetProjection(ds.GetProjection())
band = dst_ds.GetRasterBand(1)
band.SetNoDataValue(255)
band.WriteArray(output_map)
dst_ds = None