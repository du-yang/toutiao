import jieba
import gensim
import numpy as np
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
LabeledSentence = gensim.models.doc2vec.TaggedDocument

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class trainVec():
    def __init__(self):

        self.model = gensim.models.Doc2Vec(size=300,min_count=1,window=10,workers=4,alpha=0.025)

    def train(self,data,repeat=5):
        # 训练word2vec模型，供后面使用
        # :param data:要求数据是已经分好的词，array-like的数据格式
        # :param repeat: 在同一个数据上训练多少次
        # :return: 返回训练好的模型
        # dataList=np.array(data)
        self.model.build_vocab(data)
        for epoch in range(repeat):
            self.model.train(data)
            self.model.alpha -= 0.002  # decrease the learning rate
            self.model.min_alpha = self.model.alpha  # fix the learning rate, no decay
        # self.model.build_vocab(data)
        # self.model.train(data)
        return self.model


def labelizeReviews(reviews):
    labelized = []
    for i, v in enumerate(reviews):
        label = v[0]
        labelized.append(LabeledSentence(v[1].split('/'), [label]))
    return labelized


def trainData():
    with open('../data/question_info.txt') as f:
        lines = [[line[0],line[2]] for line in (line.strip().split('\t') for line in f) if line]
    with open('../data/user_info.txt') as f:
        user_info=[[line[0],line[2]] for line in (line.strip().split('\t') for line in f) if line]
    new_lines=np.concatenate((lines,user_info))
    #new_lines=[line.split('/') for line in new_lines]
    new_lines=labelizeReviews(new_lines)
    trainVecer = trainVec()
    mymodel = trainVecer.train(new_lines)
    mymodel.save('../model/docModel')
if __name__=='__main__':
    model = gensim.models.Doc2Vec.load('../model/docModel')
    print(model.docvecs.most_similar("d92fef9eeaec3031946f2067c60ca28b"))
    print(model.docvecs.similarity("d92fef9eeaec3031946f2067c60ca28b",'e7a2582ba3c022dcfd44e3b5f9b48091'))
    print(model.docvecs.similarity("d92fef9eeaec3031946f2067c60ca28b",'c8595a228cf63f363edbc7f77f1a8b09'))
    print(model.docvecs.similarity("d92fef9eeaec3031946f2067c60ca28b",'df975952f834068a1fb8b69459e7d54e'))
    print(model.docvecs.similarity("d92fef9eeaec3031946f2067c60ca28b",'adae67d1094aa5c4b18e3bcf42dbbe4b'))
    print('---------------------------------------------------------------')
    print(model.docvecs.similarity("d92fef9eeaec3031946f2067c60ca28b",'a13827766e4eb8370eeeb8de990419b7'))
    print(model.docvecs.similarity("d92fef9eeaec3031946f2067c60ca28b",'8d99ff8cb9be35c021bd53b7b631dd1c'))
    print(model.docvecs.similarity("d92fef9eeaec3031946f2067c60ca28b",'4ef9c7ecbd2d993e067afc3ec3d25b3b'))
    print(model.docvecs.similarity("d92fef9eeaec3031946f2067c60ca28b",'4ef9c7ecbd2d993e067afc3ec3d25b3b'))
    print(model.docvecs.similarity("d92fef9eeaec3031946f2067c60ca28b",'4ef9c7ecbd2d993e067afc3ec3d25b3b'))
    print(model.docvecs.similarity("d92fef9eeaec3031946f2067c60ca28b",'4ef9c7ecbd2d993e067afc3ec3d25b3b'))
    print(model.docvecs.similarity("d92fef9eeaec3031946f2067c60ca28b",'4ef9c7ecbd2d993e067afc3ec3d25b3b'))
    print(model.docvecs.similarity("d92fef9eeaec3031946f2067c60ca28b",'4ef9c7ecbd2d993e067afc3ec3d25b3b'))
    print(model.docvecs.similarity("d92fef9eeaec3031946f2067c60ca28b",'4ef9c7ecbd2d993e067afc3ec3d25b3b'))
    # for e in parameter.most_similar("1261"):
    #      print(e[0], e[1])
    # trainData()


    # with open('../data/question_info.txt') as f:
    #     lines = [[line[0], line[2]] for line in (line.strip().split('\t') for line in f) if line]
    # with open('../data/user_info.txt') as f:
    #     user_info = [[line[0], line[2]] for line in (line.strip().split('\t') for line in f) if line]
    # new_lines = np.concatenate((lines, user_info))
    # new_lines = labelizeReviews(new_lines)
    # print(new_lines[3].words)
    # print(new_lines[3].tags)