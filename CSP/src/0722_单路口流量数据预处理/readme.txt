第一步：从数据库按方向导出数据，得到0722_e_w.xlsx等4个文件
第二步：从上述4个文件中，导出各个车道的流量数据（即e_w_1.txt等20个文件）
        这一步我是用excel复制、另存的，你也可以自己写代码.
第三步：运行batch_preprocess.py，生产0722flow.xlsx，即原始流量数据
第四步：运行write_route.py， 生成rou.xml文件