
# -*- coding: GBK -*-
import os
total=os.listdir('../datasets22/labels')
for op in total:
    with open('../datasets22/labels/%s'%(op), 'r',encoding='GBK') as p:  # ��ֻ���ķ�ʽ�򿪲���ı�ԭ�ļ�����

        lines = []
        for i in p:
            lines.append(i)  # ���н��ı������б�lines��
        p.close()
        # print(lines)
        new = []

        for line in lines:  # ���б���
            p = 0  # �������ָ��
            for bit in line[0]:
                if bit == '9' or bit =='8' or bit =='7' or bit =='6':
                    pass
                else:
                    new.append(line)  # ��б�ܺ�������ݼӵ��µ�list��
                    break

    with open('../datasets22/labels/%s'%(op), 'w',encoding='GBK') as file_write:
        for var in new:
            file_write.writelines(var)  # д��
        file_write.close()




