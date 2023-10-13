
# -*- coding: GBK -*-
import os
total=os.listdir('../datasets22/labels')
for op in total:
    with open('../datasets22/labels/%s'%(op), 'r',encoding='GBK') as p:  # 以只读的方式打开不会改变原文件内容

        lines = []
        for i in p:
            lines.append(i)  # 逐行将文本存入列表lines中
        p.close()
        # print(lines)
        new = []

        for line in lines:  # 逐行遍历
            p = 0  # 定义计数指针
            for bit in line[0]:
                if bit == '9' or bit =='8' or bit =='7' or bit =='6':
                    pass
                else:
                    new.append(line)  # 将斜杠后面的内容加到新的list中
                    break

    with open('../datasets22/labels/%s'%(op), 'w',encoding='GBK') as file_write:
        for var in new:
            file_write.writelines(var)  # 写入
        file_write.close()




