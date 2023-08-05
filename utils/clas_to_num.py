import os


def class_to_num():
    clas=os.listdir('dataset/train')
    label=sorted(list(set(clas)))
    class_to_num=dict(zip(label,range(len(clas))))
    num_to_class = {v : k for k, v in class_to_num.items()}
    return class_to_num, num_to_class
