import numpy as np
import itertools
import json
import pickle


class onehot_vec():

    def __init__(self,totalWords):

        self.totalWords = totalWords
        self.vocab = self.build_vec()
        self.maxn = max(self.vocab)


    def build_vec(self):

        words = set(self.totalWords)
        wordList = [int(item) for item in words]

        return wordList

    def list2vec(self,sentence):
        '''
        Args:
            sentence:list

        Returns:
        '''

        # sentenceVec = np.empty(maxn)
        sentenceVec = np.array([0 for i in range(self.maxn+1)])


        for i in sentence:
            sentenceVec[int(i)]+=1

        return sentenceVec

    def id2vec(self,id):
        pass


def test():
    with open('../data/invited_info_train.txt') as f:
        invited = [line.strip().split() for line in f]

    with open('../data/question_info.txt') as f:
        questions = [line.strip().split() for line in f]

    with open('../data/user_info.txt') as f:
        users = [line.strip().split() for line in f]

    usr_info_dict = {line[0]: line[1:] for line in users}
    question_info_dict = {line[0]: line[1:] for line in questions}

    fu = open('../data/user_info.json', 'w')
    fq = open('../data/question_info.json', 'w')
    json.dump(usr_info_dict, fu)
    json.dump(question_info_dict, fq)
    fq.close()
    fu.close()

    # usr_raw_dict = {line[0]:line[1:] for line in users}
    # question_raw_dict = {line[0]:line[1:] for line in questions}

if __name__ == '__main__':
    question_info = json.load(open('../data/question_info.json'))
    user_info = json.load(open('../data/user_info.json'))

    userDict = json.load(open('../data/user.json'))


    question = [question_info[key][0].strip('/').split('/')+question_info[key][1].strip('/').split('/') for key in question_info]
    user = [user_info[key][0].strip('/').split('/')+user_info[key][1].strip('/').split('/') for key in user_info]
    data = question+user
    data = [i for item in data for i in item if i]

    onehot = onehot_vec(data)
    print('init finished')
    onehot_user,onehot_question = {},{}

    for item in user_info:
        xx = onehot.list2vec([i for i in user_info[item][1].split('/') if i])
        # onehot_user[item]=xx
        print(item)

    # onehot_user = {item:onehot.list2vec([i for i in user_info[item][1].split('/') if i])
    #                for item in user_info.keys()}
    # print('-----------------------------')
    # onehot_question = {item: onehot.list2vec([i for i in question_info[item][1].split('/') if i])
    #                for item in question_info}

    # i=1
    # for key in userDict:
    #     # i+=1
    #     # if i>2:break
    #     for item in userDict[key]:
    #         # q1 = question_info[item[0]][0].split('/')*5
    #         q2 = question_info[item[0]][1].strip('/').split('/')
    #         # u1 = user_info[key][0].split('/')*5
    #         u2 = user_info[key][1].strip('/').split('/')
    #         print(u2)
    #         q_vec = onehot.list2vec([i for i in q2 if i])
    #         u_vec = onehot.list2vec([i for i in u2 if i])
    #         train_x.append(u_vec-q_vec)
    #         train_y.append(int(item[-1]))

    with open('onehot_user.pkl','wb') as f:
        pickle.dump(onehot_user,f)
    x = pickle.load(open('onehot_user.pkl','rb'))
    print(x[1])


