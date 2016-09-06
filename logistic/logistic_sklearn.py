import numpy as np
import sklearn.linear_model
import pickle
import json

def load_dataset():

    with open('xx.pkl','rb') as f:
        x_train,y_train = pickle.load(f)

    return x_train,y_train

def load_dataset():
    from gensim.models import Doc2Vec
    vecModel = Doc2Vec.load('../model/docModel')
    vecModel = vecModel.docvecs
    with open('../data/user.json') as f:
        user_question = json.load(f)

    x_train ,y_train = [],[]
    for item in user_question:
        for qlist in user_question[item]:
            # x_train.append(vecModel[item]-vecModel[qlist[0]])
            x_train.append(np.concatenate((vecModel[item], vecModel[qlist[0]])))
            y_train.append(int(qlist[-1]))
            if len(y_train)>100:
                # print(y_train)
                yield x_train,y_train
                x_train, y_train = [], []


def train_logistic():

    logisticModel = sklearn.linear_model.LogisticRegression()

    i=1
    for i in range(10):
        for x_train,y_train in load_dataset():
            try:
                logisticModel.fit(x_train,y_train)
            except ValueError:
                print(ValueError)
            print('训练完第%s轮'%i)
            # print(x_train)
            print(y_train)
            print(logisticModel.predict(x_train))
            print(logisticModel.predict_proba(x_train)[:,1:].flatten())
            i+=1


    # with open('logisticModel.pkl','wb') as f:
    #     pickle.dump(logisticModel,f)

def predict(vec):

    with open('logisticModel.pkl','rb') as f:
        logisticModel = pickle.load(f)

    return logisticModel.predict(vec)

if __name__ == '__main__':
    train_logistic()
    # from gensim.models import Doc2Vec
    # vecModel = Doc2Vec.load('../model/docModel')
    # vecModel = vecModel.docvecs
    # #
    # with open('logisticModel.pkl', 'rb') as f:
    #     logisticModel = pickle.load(f)
    # with open('../data/validate_nolabel.txt') as f:
    #     to_predict = [line.strip().split(',') for line in f]
    # #
    # with open('../result/logistic.csv','w') as f:
    #     f.write(','.join(to_predict[0])+'\n')
    #     for item in to_predict[1:]:
    #         vec = vecModel[item[0]]-vecModel[item[1]]
    #         # vec = np.concatenate((vecModel[item[0]],vecModel[item[1]]))
    #         # vec.reshape(-1,1)
    #         print(item[0],item[1],logisticModel.predict_proba([vec])[0][1])
    #         f.write(','.join([item[0],item[1],str(logisticModel.predict_proba([vec])[0][1])]) + '\n')
    # # for item in to_predict[1:100]:
    # #     vec = vecModel[item[0]]-vecModel[item[1]]
    # #
    # #     print(item[0],item[1],logisticModel.predict([vec]))

