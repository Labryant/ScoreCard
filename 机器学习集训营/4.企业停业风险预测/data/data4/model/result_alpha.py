import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def avgWithAlpha(files, w, alpha = 0.21):
    '''
    files: 文件名
    alpha: 权重
    '''
    df_result = None
    for i in range(len(files)):
        df_tmp = pd.read_csv(files[i])
        if i == 0:
            df_result = df_tmp
            df_result['PROB'] = df_result['PROB'] * w[i]
        else:
            df_result['PROB'] = df_result['PROB'] + df_tmp['PROB'] * w[i]

    df_result.loc[df_result['PROB']>=alpha,'FORTARGET'] = 1
    df_result.loc[df_result['PROB']<alpha,'FORTARGET'] = 0

    return df_result

files = ['../xresult/merge_sub_file2_6.csv','../xresult/merge_sub_file1_4.csv',
         '../xresult/merge_sub_16.csv', '../xresult/merge_sub_18.csv']
w = [0.5,0.4,0.06,0.04]

df_result = avgWithAlpha(files,w)

df_result.to_csv('../xresult/merge_sub_file_with_w.csv',index=False,index_label=False)
