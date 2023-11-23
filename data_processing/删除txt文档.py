
# -*- coding: GBK -*-
import os
total=os.listdir('F:\隐裂数据集\六种缺陷数据集')

for op in total:
        total2 = os.listdir('F:\隐裂数据集\六种缺陷数据集\%s'%(op))
        for op2 in total2:
            if op2.endswith('.txt'):
                with open('F:\隐裂数据集\六种缺陷数据集\{0}\{1}'.format(op,op2), 'r', encoding='GBK') as p:
                    lines = []
                    for i in p:
                        lines.append(i)  # 逐行将文本存入列表lines中
                    p.close()
                    # print(lines)
                new = []

                for line in lines:  # 逐行遍历
                    p = 0  # 定义计数指针
                    for bit in line[0]:
                        if bit == '0' or bit =='8' or bit =='7' or bit =='6':
                            pass
                        elif bit == '9':
                            iky=line.split()
                            iky[0]="5"
                            s = ""
                            for i in iky:
                                s = s + str(i) + " "
                            s=s+'\n'
                            new.append(s)  # 将斜杠后面的内容加到新的list中
                        else:
                            iky = line.split()
                            if iky[0] == "1":
                                iky[0] = "0"
                            elif iky[0] == "2":
                                iky[0] = "1"
                            elif iky[0] == "3":
                                iky[0] = "2"
                            elif iky[0] == "4":
                                iky[0] = "3"
                            elif iky[0] == "5":
                                iky[0] = "4"
                            s = ""
                            for i in iky:
                                s = s + str(i) + " "
                            s = s + '\n'
                            new.append(s)  # 将斜杠后面的内容加到新的list中
                with open('F:\隐裂数据集\六种缺陷数据集\{0}\{1}'.format(op,op2), 'w',encoding='GBK') as file_write:
                    for var in new:
                        file_write.writelines(var)  # 写入
                    file_write.close()




