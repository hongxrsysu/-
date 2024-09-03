import os
os.environ['SDK_CLIENT_HOST'] = 'http://engine-aiearth.aliyun.com'
from aiearth import core
# 鉴权
core.Authenticate(token='cdcb64f753fe1baf18c954b1d9e9c79f')
# 获取token的网站：https://engine-aiearth.aliyun.com/#/utility/auth-token
## 运行代码：
# python /root/常态化/detect.py > /root/常态化/train.log 2>&1
# tail -f /root/常态化/train.log
from datetime import date, timedelta
import csv
import numpy as np
from aiearth import core
from aiearth.data.search import opendata_client
from aiearth.data.loader import DataLoader
from aiearth.data.stac_id import OpenStacId
from matplotlib import pyplot as plt
from datetime import datetime
from pystac import Item
from shapely import box
from shapely import Polygon
from typing import Optional, List
from typing import Union
from pyproj import CRS, Transformer
from pystac import Item
from shapely import box, Polygon
import time

def sort_items_by_intersection_area(items, bbox):
    item_and_area = list()
    for item in items:
        item_bbox = item.bbox
        item_bbox_polygon: Polygon = box(*item_bbox)
        search_bbox_polygon: Polygon = box(*bbox)
        area = item_bbox_polygon.intersection(search_bbox_polygon).area
        item_and_area.append((item, area))

    sorted_items = sorted(item_and_area, key=lambda p: p[1], reverse=True)
    return [e[0] for e in sorted_items]

def transform(item: Item, bbox: List[int], from_crs: Union[CRS, int], asset_key: str):
    """
    将地理坐标，或者投影坐标，通过使用pyproj转换为影像的行列坐标
    :param item: 影像 stac Item, 包含了影像的投影六参数
    :param bbox: 任意的bbox，同时标注from_crs; bbox只接受 (minx, miny, maxx, maxy) 格式；也即左下，右上坐标
    :param from_crs: bbox的CRS信息，可以为 pyproj.CRS, 可以为EPSG code, 如 4326
    :param asset_key: 目标波段的asset，因为波段分辨率不一致的原因，此处需要指定波段名称
    :return: 转换bbox得到的行列坐标
    """

    # transform bbox to target crs bbox
    transformer: Transformer = Transformer.from_crs(from_crs, item.properties['proj:epsg'], always_xy=True)
    transformed_bbox = transformer.transform_bounds(left=bbox[0], bottom=bbox[1], right=bbox[2], top=bbox[3])
    # calculate item bbox in target crs
    stac_transform = item.assets[asset_key].extra_fields['proj:transform']
    upper_left_x = stac_transform[2]
    pixel_width = stac_transform[0]
    upper_left_y = stac_transform[5]
    pixel_height = stac_transform[4]
    width, height = item.assets['B4'].extra_fields['proj:shape']
    bottom_right_x = upper_left_x + width * abs(pixel_width)
    bottom_right_y = upper_left_y - height * abs(pixel_height)
    item_bbox = (upper_left_x, bottom_right_y, bottom_right_x, upper_left_y)
    # check transformed-bbox and item's bbox has intersection, return intersection bbox if intersected
    transformed_bbox_polygon: Polygon = box(*transformed_bbox)
    item_bbox_polygon: Polygon = box(*item_bbox)
    intersected = transformed_bbox_polygon.intersects(item_bbox_polygon)
    if not intersected:
        return 0,0,0,0
        # raise ValueError(f"输入bbox {bbox} 转换后 {transformed_bbox} 与 item 的 bbox {item_bbox} 没有交集")
    else:
        transformed_bbox_polygon = transformed_bbox_polygon.intersection(item_bbox_polygon)
    transformed_bbox = transformed_bbox_polygon.bounds
    # use item's geoTransform and transformed-bbox to get the row-column index(offset) and size
    x_offset = abs(transformed_bbox[0] - item_bbox[0]) / abs(pixel_width)
    y_offset = abs(transformed_bbox[3] - item_bbox[3]) / abs(pixel_height)
    x_size = abs(transformed_bbox[2] - transformed_bbox[0]) / abs(pixel_width)
    y_size = abs(transformed_bbox[3] - transformed_bbox[1]) / abs(pixel_height)
    return int(x_offset), int(y_offset), int(x_size), int(y_size)

# 获取 STAC 客户端
client_opendata = opendata_client()

from mmdet.apis import init_detector, show_result_pyplot
from mmrotate.apis import inference_detector_by_patches
import mmrotate
 
config_file = '/root/mmrotate/Rmosaic.py'
checkpoint_file = '/root/mmrotate/epoch_100.pth'
model = init_detector(config_file, checkpoint_file,'cuda:0')

fishnet_txt = '/root/常态化/fishnet_coordinate.txt'
with open(fishnet_txt) as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        Left_X, Left_Y, Right_X, Right_Y = map(float, row)
        start_date = datetime(2024, 8, 24)   ## 改这里！
        end_date = datetime(2024, 8, 26)    ## 改这里！
        delta = timedelta(days=2)
        while start_date <= end_date:   
            date1 = start_date
            date2 = start_date+timedelta(days=1)
            bbox = [Left_X, Left_Y, Right_X, Right_Y]
            search_req = client_opendata.search(collections=['SENTINEL_MSIL2A'], bbox=bbox,query=['eo:cloud_cover<80'],
                                    sortby=['+eo:cloud_cover'],datetime=(date1, date2))
            time.sleep(0.01)
            req_list = list(search_req.items())
            if req_list:
                items_sorted = sort_items_by_intersection_area(req_list, bbox=bbox)  
                item = items_sorted[0]
                print(item.id)
                dataloader = DataLoader(OpenStacId(item.id)) 
                dataloader.load()
                x_offset, y_offset, x_size, y_size = transform(item, bbox, 4326, "B4")
                if x_size ==0: ## 如果不相交（数据有问题）。那么继续执行下一次for
                    break;
                img = np.ndarray(shape=(y_size, x_size, 3))
                for idx, band_name in enumerate(('B2', 'B3', 'B4')):
                    channel = dataloader.block(band_name = band_name, offset_size=(x_offset, y_offset, x_size, y_size))
                    img[:, :, idx] = channel
                img[img > 3500] = 3500     ## 拉伸影像,3500
                img[img < 0] = 0
                img = (img / 3500 * 255).astype('uint8')

                #########################################  做推理  写结果  #####################################
                result = inference_detector_by_patches(model, img, [512], [500], [1.0], 0.1)   
                det = result[0] ## 如果检测到了船，那就写结果
                if len(det)!=0 :    
                    date = item.id[11:19]
                    arr_str = np.array2string(det, separator=',', formatter={'float_kind': lambda x: "%.2f" % x})

                    # 添加字段到每一行的末尾
                    arr_str_with_fields = '\n'.join([f"{row}{Left_X},{Right_Y},{Right_X},{Left_Y},{x_size},{y_size},{date}" for row in arr_str.split('\n')])
                    # 将结果写入txt文件
                    with open('/root/常态化/aaa_plot/predict_txt/20240824_0826.txt', 'a') as f:
                        f.write(arr_str_with_fields)
                        f.write('\n')
            start_date += delta
            