global _global_dict,_global_dict_1
_global_dict = {}
_global_dict_1 = {}

def set_value(key,i,value):
    """ 定义一个全局变量 """
    _global_dict[key][i] = value

def set_value_1(key_1,value_1):
    """ 定义一个全局变量 """
    _global_dict_1[key_1] = value_1


def get_value(key,defValue=None):
    """ 获得一个全局变量,不存在则返回默认值 """

    try:
        return _global_dict[key]
    except KeyError:  # 查找字典的key不存在的时候触发
        return defValue

def get_value_1(key_1,defValue=None):
    """ 获得一个全局变量,不存在则返回默认值 """

    try:
        return _global_dict_1[key_1]
    except KeyError:  # 查找字典的key不存在的时候触发
        return defValue










