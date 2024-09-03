##############################################   数据预处理    #####################################################
import numpy as np
import cv2
from geopy import Point
from geopy.distance import distance
import math
from math import radians, cos, sin, atan2, sqrt
import re
import os
import shutil

from geographiclib.geodesic import Geodesic
import math

def txt2txt(folder,filename):
    with open(os.path.join(folder,filename), 'r') as f:
        # 逐行读取文件
        for line in f:
            # 去掉行末的换行符
            if line == '\n':
                continue;
            line = line.strip().replace("[", "").replace("]", ",")

            # 将一行文本根据空格分隔成多个字符串，存储到data列表中
            data = line.split(",")
            data =list(filter(None, data)) 
            x = float(data[0])
            y = float(data[1])
            w=float(data[2])
            h=float(data[3])
            a=float(data[4])
            k= float(data[5])
            left_x = float(data[6])
            Right_y = float(data[7])
            Right_x = float(data[8])
            left_y = float(data[9])
            xsize = int(data[10]),
            ysize = int(data[11]),
            x_len=Right_x-left_x
            y_len=Right_y-left_y           
            date = data[12]  
            area = w*h
            
            if k > 0.6 and w<48 and h<8: # 如果kexindu>0.6，且长宽不大于最大范围
                new_x=(x*x_len/xsize[0])+left_x
                new_y=Right_y-(y*y_len/ysize[0])
                year = date[0:4]
                month = date[4:6]
                day = date[6:8]
                # 保存该行数据到新的txt
                with open(txtname, 'a') as output_file:
                    output_file.write\
                    (f"{new_x} {new_y} {w} {h} {k} {a} {area} {year} {month} {day}\n")


path = '/root/常态化/aaa_plot/predict_txt'
for folder, _, filenames in os.walk(path):
    for filename in filenames:
        txtname = '/root/常态化/aaa_plot/true_coordinate/true_coordinate'+filename
        txt2txt(folder, filename) #运行前需要删掉true_coirdinate开头的那些文件
		
##############################################  文字转矢量  #######################################
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def txt2shp(filename):
    # 读取文本文件，并指定列名
    file_name =path+filename

    df = pd.read_csv(file_name, sep=' ', header=None, 
                     names=['x', 'y', 'w', 'h', 'k', 'angle', 'area', 'year', 'month', 'day'])

    # 将中心点坐标转换为Point对象
    geometry = [Point(row['x'], row['y']) for _, row in df.iterrows()]

    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(df[['x', 'y', 'w', 'h', 'area', 'k','angle','year', 'month', 'day']], geometry=geometry, crs='EPSG:4326')
    new_name=filename.split(".")
    new_name=new_name[0]
    new_name=new_name[15:]
    # 保存为shapefile
    gdf.to_file('/root/常态化/aaa_plot/shp/'+new_name+'.shp')

path = '/root/常态化/aaa_plot/true_coordinate/'
for folder, _, filenames in os.walk(path):
    for filename in filenames:
        txt2shp(filename) #运行前需要删掉true_coirdinate开头的那些文件