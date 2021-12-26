import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC

from osgeo import gdal

attributes = ['Band 1', 'Band 2', 'Band 3', 'NDVI', 'NDWI']

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

train=pd.read_csv("./train.csv")
X = train.filter(attributes)
y = train.filter(['Label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)

clf = RandomForestClassifier(n_estimators=200, max_depth= 14, min_samples_split= 4)

# clf = KNeighborsClassifier(leaf_size=30)

# clf = MLPClassifier()

# clf = SVC()

clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print('Accuracy score - Test dataset: {}'.format(accuracy_score(y_test, y_pred)))
print('F1 score - Test dataset: {}'.format(f1_score(y_test, y_pred, average= None)))

ds = gdal.Open(r'.\dataImage.tif', gdal.GA_ReadOnly)
data = ds.ReadAsArray()
bands, rows, cols = data.shape
print(rows)
print(cols)

rsl = []
for i in range(rows):
    for j in range(cols):
        if data[0, i, j] > -9999:
            tmp = np.append([i, j], data[:, i, j])
            rsl.append(tmp)

df = pd.DataFrame(rsl, columns=['rows', 'cols', 'Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5', 'Band 6', 'Band 7'])

df.to_csv('map.csv', index=False)

map_df = pd.read_csv(r'.\map.csv')

map_df["NDVI"] = (map_df["Band 4"] - map_df["Band 1"])/(map_df["Band 4"] + map_df["Band 1"])
map_df["NDWI"] = (map_df["Band 2"] - map_df["Band 4"])/(map_df["Band 4"] + map_df["Band 2"])

tmp = map_df.filter(attributes)
map_df['pred'] = clf.predict(tmp)
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