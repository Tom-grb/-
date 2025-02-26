import random
import pandas as pd
import numpy as np
import copy
import math
import matplotlib.pyplot as plt


def init_data(class_data, classroom_data, course_teacher_data):
    # 初始化数据
    empty_time = []
    for w in range(16):
        for d in range(5):
            for j in range(5):
                empty_time.append([w, d, j])

    empty_class = dict()
    for index, row in class_data.iterrows():
        empty_class[row['班级名称']] = empty_time.copy()
        # ！！！要使用.copy()，不然empty_classroom, empty_teacher, empty_class，其中一个表变化，其他也会变化，字典分配到的应该是地址，所以共享一个empty_time

    teacher_list = ['T001', 'T002', 'T003', 'T004', 'T005', 'T006', 'T007', 'T008', 'T009', 'T009', 'T010', 'T011',
                    'T012', 'T013', 'T014', 'T015', 'T016', 'T017', 'T018', 'T019', 'T020'
                    ]
    empty_teacher = dict()
    for teacher_name in teacher_list:
        empty_teacher[teacher_name] = empty_time.copy()

    empty_classroom = dict()
    for index, row in classroom_data.iterrows():
        empty_classroom[row['教室']] = empty_time.copy()

    # 创建课表模板
    # 周 日 节 教室
    matrix = np.full((16, 5, 5, 26), fill_value=None)

    return empty_classroom, empty_teacher, empty_class, matrix


def Less_than_six_hours(matrix, week, day, joint, name, cort):
    # cort 判断班级还是老师

    time = 0
    for k in range(joint):
        for c in range(26):
            if matrix[week][day][k][c] is None:
                continue
            elif matrix[week][day][k][c][cort] == name:
                time += 2

    if time > 5:
        return True
    else:
        return False


# 导入数据 class_data, classroom_data, course_teacher_data / matrix 课表 / empty_class 班级的空闲时间段 / empty_teacher / empty_classroom 教室空闲时间段
def cartesian_product(class_data, classroom_data, course_teacher_data, matrix, empty_class, empty_teacher,
                      empty_classroom):
    school_timetable = []

    for week in range(16):
        for day in range(5):
            for joint in range(5):
                for j in range(9):
                    # 循环9次，用来随机分配教室，如果太大，会集中分配在前半学期
                    if len(class_data) == 0:
                        continue
                    # 从节开始选择，随机选择一个班级和课程，然后匹配老师
                    random_row = class_data.sample(n=1)
                    course_name = random_row['课程名称'].item()
                    class_name = random_row['班级名称'].item()

                    cur_time = [week, day, joint]

                    teacher_list = course_teacher_data[course_teacher_data['课程名称'] == course_name]

                    # 查找这时间段有没有相同的班级
                    if cur_time not in empty_class[class_name]:
                        # 班级这时间段没时间
                        continue

                    # 判断班级上课时间是否超过6学时
                    if Less_than_six_hours(matrix, week, day, joint, class_name, 0):
                        continue

                    # 选择老师
                    teacher_name = ''
                    if course_name != '人工智能基础':
                        # 横向遍历,查看这个时间段是否有冲突
                        if cur_time in empty_teacher[teacher_list['教师'].item()]:
                            teacher_name = teacher_list['教师'].item()
                            # 纵向遍历,查看这个是否老师教学时间是否超过6学时
                            if Less_than_six_hours(matrix, week, day, joint, teacher_name, 1):
                                continue
                        elif pd.notna(teacher_list['教师.1'].item()):
                            if cur_time in empty_teacher[teacher_list['教师.1'].item()]:
                                teacher_name = teacher_list['教师.1'].item()
                                # 纵向遍历,查看这个是否老师教学时间是否超过6学时
                                if Less_than_six_hours(matrix, week, day, joint, teacher_name, 1):
                                    continue
                            else:
                                continue
                        else:
                            continue

                    else:
                        tList = ['T001', 'T002', 'T003', 'T004', 'T005', 'T006', 'T007', 'T008', 'T009', 'T010', 'T011',
                                 'T012', 'T013', 'T014', 'T015', 'T016', 'T017', 'T018', 'T019', 'T020']

                        # 找出可以上人工智能基础的老师

                        for (index, name) in enumerate(tList.copy()):  # 要使用copy，否则会改变tList，就不会遍历所有教师了
                            # 横向遍历
                            if cur_time not in empty_teacher[name]:
                                tList.remove(name)

                            # 纵向遍历
                            if Less_than_six_hours(matrix, week, day, joint, name, 1):
                                tList.remove(name)

                        if len(tList) == 0:
                            continue
                        teacher_name = random.choice(tList)

                    # 判断教室是否符合
                    for classroomId in range(26):
                        RL = classroom_data.iloc[classroomId]['教室'][0]
                        if matrix[week][day][joint][classroomId] is not None:
                            continue
                        # 判断教室是否合适，容量和实验学时还是讲课学时
                        if classroom_data.iloc[classroomId]['容量'] < random_row['排课人数'].item():
                            continue
                        elif random_row['实验学时'].item() == 0 and random_row['讲课学时'].item() == 0:
                            continue
                        elif RL == 'R' and random_row['讲课学时'].item() == 0:
                            continue
                        elif RL == 'L' and random_row['实验学时'].item() == 0:
                            continue

                        classroom_name = classroom_data.iloc[classroomId]['教室']

                        # 根据是R还是L，减学时
                        if RL == 'R':
                            class_data.loc[(class_data['课程名称'] == course_name) & (
                                    class_data['班级名称'] == class_name), '讲课学时'] -= 2
                        else:
                            class_data.loc[(class_data['课程名称'] == course_name) & (
                                    class_data['班级名称'] == class_name), '实验学时'] -= 2

                        # 查找到教室，就修改matrix对应位置为 “班级名称”、“老师名字” ，标记这个教室在这个是否被占用
                        matrix[week][day][joint][classroomId] = [class_name, teacher_name, course_name]

                        # 将[周次、周几、第几节、课程名称、班级名称、老师]存储到全局数组
                        school_timetable.append(
                            [week, day, joint, course_name, classroom_name, class_name, teacher_name])

                        # 查看是有学时都学完的
                        indexes_to_drop = class_data[
                            (class_data['讲课学时'] <= 0) & (class_data['实验学时'] <= 0)].index
                        class_data.drop(indexes_to_drop, inplace=True)

                        # 删除老师，班级，教室这时间段的空闲时间
                        # print(j, cur_time, course_name, class_name, teacher_name, classroom_name)
                        empty_teacher[teacher_name].remove(cur_time)
                        empty_class[class_name].remove(cur_time)
                        empty_classroom[classroom_name].remove(cur_time)

                        break

    # print(class_data)
    # 将还有学时的班级进行分配
    # 找到教室,老师,班级同时都有空的时间点
    while len(class_data)>0:
        for row in copy.deepcopy(class_data).iterrows():
            class_name = row[1]['班级名称']
            course_name = row[1]['课程名称']
            room_time = row[1]['讲课学时']
            lab_time = row[1]['实验学时']

            # 找老师
            tt = course_teacher_data[course_teacher_data['课程名称'] == course_name]
            teacher_name = ''
            if pd.isna(tt['教师.1'].item()) and tt['教师'].item() != '所有教师':
                teacher_name = tt['教师'].item()
            elif tt['教师'].item() != '所有教师' and pd.notna(tt['教师.1'].item()):
                n1 = tt['教师'].item()
                n2 = tt['教师.1'].item()
                if len(empty_teacher[n1]) > len(empty_teacher[n2]):
                    teacher_name = n1
                else:
                    teacher_name = n2
            elif tt['教师'].item() == '所有教师':
                # 找空闲时间最长的
                l = 0
                for i in range(1, 21):
                    t_name = f"T{i:03}"
                    if l < len(empty_teacher[t_name]):
                        teacher_name = t_name

            common_time1 = [item for item in empty_teacher[teacher_name] if item in empty_class[class_name]]  # 老师和班级都空闲的时间
            # 找教室
            if room_time > 0:
                for (id, classroom_row) in enumerate(classroom_data.iterrows()):
                    if classroom_row[1][0][0] != 'R':
                        break
                    else:
                        empty_classroom_name = empty_classroom[classroom_row[1][0]]  # 目前教室的空闲时间
                        common_time2 = [item for item in common_time1 if item in empty_classroom_name]  # 三者都空闲的时间
                        ok_time = []
                        for free_time in common_time2:
                            if Less_than_six_hours(matrix, free_time[0], free_time[1], free_time[2], class_name,
                                                   0) or Less_than_six_hours(matrix, free_time[0], free_time[1],
                                                                             free_time[2], teacher_name, 1):
                                continue
                            else:
                                ok_time = free_time
                                break
                        if len(ok_time) == 0:
                            continue
                        # print(ok_time)
                        class_data.loc[(class_data['课程名称'] == course_name) & (class_data['班级名称'] == class_name), '讲课学时'] -= 2
                        matrix[ok_time[0]][ok_time[1]][ok_time[2]][id] = [class_name, teacher_name, course_name]
                        school_timetable.append(
                            [ok_time[0], ok_time[1], ok_time[2], course_name, classroom_row[1][0], class_name,
                             teacher_name])
                        empty_class[class_name].remove(ok_time)
                        empty_classroom[classroom_row[1][0]].remove(ok_time)
                        empty_teacher[teacher_name].remove(ok_time)
                        common_time1.remove(ok_time)  # 删除老师交班级的这时间段
                        break


            # 找实验室
            if lab_time > 0:
                for (id, classroom_row) in enumerate(classroom_data.iterrows()):
                    if classroom_row[1][0][0] == 'R':
                        continue
                    else:
                        empty_classroom_name = empty_classroom[classroom_row[1][0]]  # 目前教室的空闲时间
                        common_time2 = [item for item in common_time1 if item in empty_classroom_name]  # 三者都空闲的时间
                        ok_time = []
                        for free_time in common_time2:
                            if Less_than_six_hours(matrix, free_time[0], free_time[1], free_time[2], class_name,
                                                   0) or Less_than_six_hours(matrix, free_time[0], free_time[1],
                                                                             free_time[2], teacher_name, 1):
                                continue
                            else:
                                ok_time = free_time
                                break
                        if len(ok_time) == 0:
                            continue
                        class_data.loc[(class_data['课程名称'] == course_name) & (class_data['班级名称'] == class_name), '实验学时'] -= 2
                        matrix[ok_time[0]][ok_time[1]][ok_time[2]][id] = [class_name, teacher_name, course_name]
                        school_timetable.append(
                            [ok_time[0], ok_time[1], ok_time[2], course_name, classroom_row[1][0], class_name,
                             teacher_name])
                        empty_class[class_name].remove(ok_time)
                        empty_classroom[classroom_row[1][0]].remove(ok_time)
                        empty_teacher[teacher_name].remove(ok_time)
                        common_time1.remove(ok_time)  # 删除老师交班级的这时间段
                        break

        indexes_to_drop = class_data[
            (class_data['讲课学时'] <= 0) & (class_data['实验学时'] <= 0)].index
        class_data.drop(indexes_to_drop, inplace=True)

        # print(len(class_data))


    return matrix, empty_class, empty_teacher, empty_classroom, school_timetable


def soft_constraint(empty_class, empty_teacher):
    un_list = []  # 存储连堂课
    # 有老师或者班级连堂课的加一
    fitness = 0
    for class_name in class_data['班级名称']:
        for w in range(16):
            for d in range(5):
                for j in range(4):
                    time1 = [w, d, j]
                    time2 = [w, d, j + 1]
                    if time1 not in empty_class[class_name] and time2 not in empty_class[class_name]:
                        fitness += 1

    teacher_list = ['T001', 'T002', 'T003', 'T004', 'T005', 'T006', 'T007', 'T008', 'T009', 'T009', 'T010', 'T011',
                    'T012', 'T013', 'T014', 'T015', 'T016', 'T017', 'T018', 'T019', 'T020'
                    ]
    for teacher_name in teacher_list:
        for w in range(16):
            for d in range(5):
                for j in range(4):
                    time1 = [w, d, j]
                    time2 = [w, d, j + 1]
                    if time1 not in empty_teacher[teacher_name] and time2 not in empty_teacher[teacher_name]:
                        fitness += 1
    return fitness


def show(matrix):
    for week in range(16):
        for day in range(5):
            print(f"第{week}周 星期{day} \t", end='')
            for index, row in classroom_data.iterrows():
                print(f"{row['教室']}", end='\t')  # 假设教室名称不超过20个字符
            print()

            # 打印课程表
            for joint in range(5):
                print(f"第{joint}节课", end='\t\t')
                for classroomId in range(26):
                    # 获取当前单元格的内容
                    content = matrix[week][day][joint][classroomId]
                    if content is None:
                        content_str = "None"
                        content_str = f"{content_str}"
                    else:
                        content_str = f"{content[0] + content[1]:<30}"  # 假设每个单元格的内容不超过20个字符
                    print(f"{content_str}\t", end='')
                print()
            print()


def variation(matrix, school_timetable, empty_class, empty_teacher, empty_classroom):
    change_matrix = copy.deepcopy(matrix)
    change_school_timetable = copy.deepcopy(school_timetable)
    change_empty_class = copy.deepcopy(empty_class)
    change_empty_teacher = copy.deepcopy(empty_teacher)
    change_empty_classroom = copy.deepcopy(empty_classroom)
    for _ in range(150):
        info = random.choice(
            change_school_timetable)  # [week, day, joint, course_name, classroom_name, class_name, teacher_name]
        # 抽出来,放到15-16周
        # 随机选择时间,地点(如果是lab就选回lab,room同理)
        rand_time = [random.randint(5, 16), random.randint(0, 5), random.randint(0, 5)]
        classroom_name = ''
        if info[5][0] == 'R':
            for name in empty_classroom:
                if rand_time in empty_classroom[name] and name[0] == 'R':
                    classroom_name = name
                    break
        else:
            for name in change_empty_classroom:
                if rand_time in change_empty_classroom[name] and name[0] == 'L':
                    classroom_name = name
                    break

        if classroom_name == '':
            # 没找到教室
            continue

        # 判断老师是否有时间
        if rand_time not in change_empty_teacher[info[6]]:
            continue

        # 判断班级是否有空
        if rand_time not in change_empty_class[info[5]]:
            continue

        # 判断教室是否有空
        if rand_time not in change_empty_classroom[info[4]]:
            continue

        # 纵向判断班级
        if Less_than_six_hours(change_matrix, rand_time[0], rand_time[1], rand_time[2], info[5], 0):
            continue

        # 纵向判断老师
        if Less_than_six_hours(change_matrix, rand_time[0], rand_time[1], rand_time[2], info[6], 1):
            continue

        # 都可以就变异
        cid1 = classroom_data.index[classroom_data['教室'] == info[4]].tolist()[0]
        cid2 = classroom_data.index[classroom_data['教室'] == classroom_name].tolist()[0]
        change_matrix[info[0]][info[1]][info[2]][cid1] = None
        change_matrix[rand_time[0]][rand_time[1]][rand_time[2]][cid2] = [info[5], info[6], info[
            3]]  # [class_name, teacher_name, course_name]
        change_school_timetable.remove(info)
        change_school_timetable.append([info[0], info[1], info[2], info[3], classroom_name, info[5], info[6]])
        # 删除空闲时间
        change_empty_class[info[5]].remove(
            rand_time)  # [week, day, joint, course_name, classroom_name, class_name, teacher_name]
        change_empty_teacher[info[6]].remove(rand_time)
        change_empty_classroom[info[4]].remove(rand_time)
        # 添加空闲时间
        change_empty_class[info[5]].append(info[:3])
        change_empty_teacher[info[6]].append(info[:3])
        change_empty_classroom[classroom_name].append(info[:3])

    return change_matrix, change_school_timetable, change_empty_class, change_empty_teacher, change_empty_classroom


# 导入数据
class_data = pd.read_excel('class_data.xlsx', engine='openpyxl')
classroom_data = pd.read_excel('classroom_data.xlsx', engine='openpyxl')
course_teacher_data = pd.read_excel('course_teacher_data.xlsx', engine='openpyxl')

# 初始化数据 empty_class 班级的空闲时间段 / empty_teacher / empty_classroom 教室空闲时间段
empty_classroom, empty_teacher, empty_class, matrix = init_data(class_data, classroom_data, course_teacher_data)

show(matrix)

# 退火算法
# 温度
T = 3.5
generation = 0
min_fitness = 100000
best_matrix = list()
best_school_timetable = list

# 创建课表
new_matrix, new_empty_class, new_empty_teacher, new_empty_classroom, new_school_timetable = cartesian_product(
    class_data, classroom_data, course_teacher_data, matrix, empty_class,
    empty_teacher,
    empty_classroom)

# np.save('school_timetable.npy', new_school_timetable) # 存储到本地

# 计算软约束
fitness = soft_constraint(new_empty_class, new_empty_teacher)
if fitness < min_fitness:
    min_fitness = fitness
    best_matrix = copy.deepcopy(new_matrix)
    best_school_timetable = copy.deepcopy(new_school_timetable)


fitness_list = []
min_fitness_list = []

while T > 0.1:

    matrix2, school_timetable2, empty_class2, empty_teacher2, empty_classroom2 = variation(
        new_matrix, new_school_timetable, new_empty_class,
        new_empty_teacher, new_empty_classroom)

    # 求软适应度
    fitness = soft_constraint(empty_class2, empty_teacher2)
    # print('now_fitness:', fitness, ' min_fitness:', min_fitness)
    fitness_list.append(fitness)
    min_fitness_list.append(min_fitness)
    # 判断是否接收新解
    if fitness < min_fitness or math.exp(-(fitness - min_fitness) / T) > random.random():
        min_fitness = fitness
        best_matrix = copy.deepcopy(new_matrix)
        best_school_timetable = copy.deepcopy(new_school_timetable)
        new_matrix, new_school_timetable, new_empty_class, new_empty_teacher, new_empty_classroom = matrix2, school_timetable2, empty_class2, empty_teacher2, empty_classroom2

    T -= 0.02

show(best_matrix)
np.save('school_timetable.npy', best_school_timetable) # 存储到本地
np.save('matrix.npy', best_matrix) # 存储到本地
x = list(range(len(min_fitness_list)))
# 创建折线图
plt.plot(x, min_fitness_list, label='min', color='red', linestyle='-', marker='o', linewidth=1, markersize=3)
plt.plot(x, fitness_list, label='now', color='blue', linestyle='-', marker='x', linewidth=1, markersize=3)

# 添加标题和标签
plt.xlabel('number of iterations')
plt.ylabel('Soft fitness')

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
