# -
# 论文《基于RShipDet的南沙群岛船只时空分异特征》的补充代码。包括影像自动获取，船只检测，和船只检测结果矢量化。
# 首先先注册AI Earth，同时和作者联系获得模型；随后运行detect.py；
python /root/常态化/detect.py > /root/常态化/train.log 2>&1

tail -f /root/常态化/train.log
