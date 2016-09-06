import json
import pickle
import numpy.random as nrg
import numpy as np


def bayes_parameter():
    question = json.load(open('../data/question.json'))
    user = json.load(open('../data/user.json'))

    with open('p_q_u_qu.pkl','wb') as f:
        p_q = {}
        p_u = {}
        p_uq = {}
        for key in question:
            if len(question[key]) >0:
                isAnswer = [float(line[-1]) for line in question[key]]
                p = sum(isAnswer)/len(isAnswer)
                p_q[key] = p if p>0.05 else nrg.uniform(0.0002,0.05)+p

        for ukey in user:
            if len(user[ukey]) > 0:
                tags = set([line[-2] for line in user[ukey]])
                tag2answer = {tag:[] for tag in tags}
                for line in user[ukey]:
                    tag2answer[line[-2]].append(int(line[-1]))
                for key in tag2answer:
                    p = sum(tag2answer[key])/len(tag2answer[key])
                    tag2answer[key] = p if p > 0.05 else nrg.uniform(0.0002, 0.05)+p
                p_uq.update({ukey:tag2answer})

                isAnswer = [float(line[-1]) for line in user[ukey]]
                p = sum(isAnswer) / len(isAnswer)

                p_u[ukey] = p if p > 0.05 else nrg.uniform(0.0002, 0.05)+p

        data=(p_q,p_u,p_uq)
        print(p_q)
        pickle.dump(data,f)

def bayes(qid,uid,vec_model,p_q_dict, p_u_dict, p_uq_dict,q2tag):
    if qid not in p_q_dict:
        for item in vec_model.docvecs.most_similar(qid,topn=200):
            if item[0] in p_q_dict:
                p_qid = p_q_dict[item[0]]
                break
    else:p_qid = p_q_dict[qid]

    if uid not in p_u_dict or q2tag[qid] not in p_uq_dict[uid]:
        for item in vec_model.docvecs.most_similar(uid, topn=200):
            if item[0] in p_u_dict and q2tag[qid] in p_uq_dict[item[0]]:
                p_uid = p_u_dict[item[0]]
                p_uqid = p_uq_dict[item[0]][q2tag[qid]]
                break
    else:
        try:
            p_uid = p_u_dict[uid]
            p_uqid = p_uq_dict[uid][q2tag[qid]]
        except Exception as e:
            print(e)
            print(p_uq_dict[uid])
            p_uqid = nrg.uniform(0,1)
            print('p_uqid',p_uqid)
    try:
        p_b = 0.5*p_uid+0.8*p_uqid+0.5*p_qid
    except:
        p_b=0
    return p_b if p_b<1 else 1.

def bayes_simple(qid,uid,vec_model, p_uq_dict,q2tag):
    flag = 2
    if uid not in p_uq_dict or q2tag[qid] not in p_uq_dict[uid]:
        flag += 1
        for item in vec_model.docvecs.most_similar(uid, topn=200):
            if item[0] in p_uq_dict and q2tag[qid] in p_uq_dict[item[0]]:
                p = p_uq_dict[item[0]][q2tag[qid]]
                flag = 1
                break
    else:
        p = p_uq_dict[uid][q2tag[qid]]

    if flag == 1 or flag ==2:
        pass
    else:
        print('-------------------------')
        p = np.abs(vec_model.docvecs.similarity(qid,uid)*0.1)

    return p



if __name__ == '__main__':
    from gensim.models.doc2vec import Doc2Vec
    vec_model = Doc2Vec.load('../model/docModel')
    p_q_dict, p_u_dict, p_uq_dict = pickle.load(open('p_q_u_qu.pkl', 'rb'))
    q2tag = json.load(open('../data/question2tag.json', 'r'))
    # p = bayes_simple('cdbe2a8e3e0fb67bf6b54e1e9dc386d9', '43d2b7d6858ed613d6f7ae03f374ee48', vec_model, p_uq_dict,q2tag)
    with open('../data/validate_nolabel.txt') as f:
        line = [line.strip().split(',') for line in f]

        result =[]
        result.append(','.join(line[0]))

        for item in line[1:]:
            # p=bayes_simple(item[0], item[1], vec_model, p_uq_dict,q2tag)
            p=bayes(item[0], item[1], vec_model, p_q_dict, p_u_dict, p_uq_dict,q2tag)
            print(item[0], item[1],p)
            result.append(','.join([item[0],item[1],str(p)]))

        with open('../result/bayes.csv','w',encoding='utf8') as w:
            for line in result:
                w.write(line+'\n')
    # bayes_parameter()
