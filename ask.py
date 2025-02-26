import numpy as np
import pandas as pd

school_timetable = np.load('school_timetable.npy')
class_data = pd.read_excel('class_data.xlsx', engine='openpyxl')

print('需要查询什么')
print('1 班级')
print('2 教师')
print('3 教室')
print('4 整个课表')
what = int(input())

# 23物联网工程班

school_timetable = sorted(school_timetable, key=lambda x: (x[0], x[1], x[2]))

if what == 1:
    name = input('输入班级名称:\n')
    # 查看这班级的总学时
    total_time = 0
    name_list = class_data[class_data['班级名称']==name]
    for (index,item) in name_list.iterrows():
        total_time+=(item['实验学时']+item['讲课学时'])

    j = 0
    for item in school_timetable:
        if item[5] == name:
            print(f'第周{int(item[0]) + 1}，周{int(item[1]) + 1},第{int(item[2]) + 1}节,{item[3]},{item[4]},{item[6]}')
            j+=1

    print('已上完所有学时' if j*2==total_time else '还有学时没上完')

elif what == 2:
    name = input('输入教师名称')
    for item in school_timetable:
        if item[6] == name:
            print(f'第周{int(item[0]) + 1}，周{int(item[1]) + 1},第{int(item[2]) + 1}节,{item[3]},{item[4]},{item[5]}')

elif what == 3:
    name = input('输入教室名称')
    for item in school_timetable:
        if item[4] == name:
            print(f'第周{int(item[0]) + 1}，周{int(item[1]) + 1},第{int(item[2]) + 1}节,{item[3]},{item[5]},{item[6]}')

else:
    print('请输入1-3')