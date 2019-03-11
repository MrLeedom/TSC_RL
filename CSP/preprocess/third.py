#!/usr/bin/python3.6
# -*- encoding: utf-8 -*-
'''
   @Author:leedom

   Created on Tue Jan 22 13:37:53 2019
   Description:处理excel文件,将其转换成csv格式
   License: (C)Copyright 2019
'''
import xlrd   #读取excel
import xlwt   #写入excel
from datetime import date,datetime

def read_excel(name):
    #打开文件
    workbook = xlrd.open_workbook('../data/' + name + '.xlsx')
    #获取所有sheet
    # print(workbook.sheet_names())   #只有一张表
    sheet_name = workbook.sheet_names()[0]

    #根据sheet索引或者名称获取sheet内容
    sheet = workbook.sheet_by_index(0)   #sheet索引从0开始
    # sheets = workbook.sheet_by_name('Sheet1')

    # sheet的名称,行数,列数
    # print(sheet.name,sheet.nrows,sheet.ncols)

    #获取整行, 整列的值(数组)
    # rows = sheet.row_values(1)  #获取第二行的内容
    f = open('../data/' + name + '.csv','w+')
    string = ''
    for k in range(sheet.nrows):
        rows = sheet.row_values(k)
        # print(rows)
        for i in range(sheet.ncols):
            if i == 0:
                if k == 0:
                    string = str(rows[i])
                else:
                    string = str(int(rows[i]))
                    
            else:
                if k == 0:
                    string += ',' + str(rows[i])      
                else:
                    string += ',' + str(int(rows[i]))    
        print(string, file = f)
        string = ''
    # cols = sheet.col_values(2)  #获取第三列的内容
    # print('rows:',rows)
    # print('cols:',cols)

    #获取单元格内容
    # print(sheet.cell(0,0).value)
    # print(sheet.cell(0,0).value.encode('utf-8'))
    #获取单元格内容的数据类型
    # print(sheet.cell(1,0).ctype)
if __name__ == "__main__":
    roads = ['airport','lihua','zhenning','jianshe4','jianshe3','jianshe2','jianshe1']
    for i in range(len(roads)):
        read_excel(roads[i])