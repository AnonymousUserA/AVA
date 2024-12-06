#!/bin/bash
# **************************************************************
# @Author      : Xiangtao Meng
# @File name   : baidu.sh
# @Project     : code
# @CreateTime  : 2022/10/24 下午4:09:56
# @Version     : v1.0
# @Description : ""
# @Update      : [序号][日期YYYY-MM-DD] [更改人姓名][变更描述]
# @Copyright © 2020-2021 by Xiangtao Meng, All Rights Reserved
# **************************************************************

eval "$(conda shell.bash hook)"
source activate evaluation

image_path=$1
cd /8T/work/DeepFake/code/util
xx=`python reptile.py --image_path $image_path`
echo $xx