from numpy.lib.function_base import median
from osgeo import gdal
import numpy as np

data_path =r'C:\Users\bangphuonglaptop\Desktop\python\dataImage.tif'

raster = gdal.Open(data_path, gdal.GA_ReadOnly)

bands = raster.RasterCount

for band in range(1, bands+1):
    data = raster.GetRasterBand(band).ReadAsArray().astype('float')
    median = np.median(data[data > 0]) 
    print("Band %s: Median = %s" % (band, round(median, 8)))
