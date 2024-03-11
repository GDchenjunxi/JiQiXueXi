import pandas as pd
import glob
import os
def get(elem):
    return elem[1]
label_path=glob.glob(r'C:\Users\Chan2\Desktop\yolov5-master_jiejing\yolov5-master\runs\detect\exp6\labels\*.txt') #('labels/*.txt')
label_path.sort()
df_submit = pd.read_csv('C:/Users/Chan2/Desktop/yolov5-master_jiejing/mchar_sample_submit_A.csv')
df_submit.set_index('file_name')
for x in label_path:
    text=open(x,'r')
    result_list=[]
    for line in text.readlines():
        result_list.append((line.split(' ')[0],line.split(' ')[1]))
    result_list.sort(key=get)
    result=''
    for j in result_list:
        result+=j[0]
    label_path=x.split('\\')[-1].split('.')[0]+'.png'
    df_submit['file_code'][df_submit['file_name']==label_path]=result
    text.close()
df_submit.to_csv('C:/Users/Chan2/Desktop/yolov5-master_jiejing/submit.csv', index=None)