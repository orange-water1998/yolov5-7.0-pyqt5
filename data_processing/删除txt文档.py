
# -*- coding: GBK -*-
import os
total=os.listdir('F:\�������ݼ�\����ȱ�����ݼ�')

for op in total:
        total2 = os.listdir('F:\�������ݼ�\����ȱ�����ݼ�\%s'%(op))
        for op2 in total2:
            if op2.endswith('.txt'):
                with open('F:\�������ݼ�\����ȱ�����ݼ�\{0}\{1}'.format(op,op2), 'r', encoding='GBK') as p:
                    lines = []
                    for i in p:
                        lines.append(i)  # ���н��ı������б�lines��
                    p.close()
                    # print(lines)
                new = []

                for line in lines:  # ���б���
                    p = 0  # �������ָ��
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
                            new.append(s)  # ��б�ܺ�������ݼӵ��µ�list��
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
                            new.append(s)  # ��б�ܺ�������ݼӵ��µ�list��
                with open('F:\�������ݼ�\����ȱ�����ݼ�\{0}\{1}'.format(op,op2), 'w',encoding='GBK') as file_write:
                    for var in new:
                        file_write.writelines(var)  # д��
                    file_write.close()




