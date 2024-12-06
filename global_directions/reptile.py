# -*- coding: utf-8 -*-

# **************************************************************
# @Author      : Xiangtao Meng
# @File name   : reptile.py
# @Project     : code
# @CreateTime  : 2022/10/23 下午10:11:31
# @Version     : v1.0
# @Description : ""
# @Update      : [序号][日期YYYY-MM-DD] [更改人姓名][变更描述]
# @Copyright © 2020-2021 by Xiangtao Meng, All Rights Reserved
# **************************************************************

from selenium import webdriver
import argparse
import warnings
warnings.filterwarnings("ignore")
chromeOptions = webdriver.ChromeOptions()

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--image_path', type=str, default = "")
args = parser.parse_args()

# 设置代理
chromeOptions.add_argument("--proxy-server=https://127.0.0.1:7890")
chromeOptions.add_argument("--proxy-server=http://127.0.0.1:7890")
browser = webdriver.Chrome(chrome_options = chromeOptions)
browser.get('https://www.duckduckgoose.ai/demo')
element_input = browser.find_element("id", "fileElem")

# 模拟文件上传
element_input.send_keys(args.image_path)

# 模拟点击
browser.find_element("id", "analyzeButton").click()

while(True):
    fake_score = browser.find_element("id", "fakeProb")
    if fake_score.text != "":
        break
xx = fake_score.text[:-1]
score = float(fake_score.text[:-1])
print(score/100.0)
browser.close()