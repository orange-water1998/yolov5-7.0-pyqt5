import yaml
import argparse
#1233

# def updata_yaml(k,v,yaml_path):
#     old_data=read_yaml_all(yaml_path) #��ȡ�ļ�����
#     old_data[k]=v #�޸Ķ�ȡ�����ݣ�k���ھ��޸Ķ�Ӧֵ��k�����ھ�����һ���ֵ�ԣ�
#     with open(yaml_path, "w") as f:
#         yaml.dump(old_data,f)
# #
# def read_yaml_all(yaml_path):
#     try:
#         # ���ļ�
#         with open(yaml_path,"r",encoding='utf8',errors='ignore') as f:
#             data=yaml.load(f,Loader=yaml.FullLoader)
#             return data
#     except:
#         return None
def get_classes(classes_path):
    with open(classes_path, encoding='GBK') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--yaml_path', default='malairisheng.yaml', type=str, help='input yaml_path')
# parser.add_argument('--classes2_path', default='../datasets/labels/classes.txt', type=str, help='input classes_path')

if __name__ == '__main__':
    # opt = parser.parse_args()
    # classes2 = opt.classes_path2
    # yaml_path = opt.yaml_path
    classes , _ = get_classes('../datasets/labels/classes.txt')
    # updata_yaml('names',classses,yaml_path)
