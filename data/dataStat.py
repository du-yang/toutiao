import json

def test():
    with open('invited_info_train.txt') as f:
        invited = [line.strip().split() for line in f]

    with open('question_info.txt') as f:
        questions = [line.strip().split() for line in f]

    with open('user_info.txt') as f:
        users = [line.strip().split() for line in f]
    print(invited[1])
    print(questions[1])
    print(users[1])

    usr_raw_dict = {line[0]:line[1:] for line in users}
    question_raw_dict = {line[0]:line[1:] for line in questions}




    fu = open('user.json','w')
    fq = open('question.json', 'w')
    userDict = {}
    questionDict = {}
    for q,u,isAswer in invited:
        if u not in userDict:
            userDict[u]=[]
            userDict[u].append([ q,usr_raw_dict[u][0],question_raw_dict[q][0],isAswer ])
        else:
            userDict[u].append([q, usr_raw_dict[u][0],question_raw_dict[q][0], isAswer])

        if q not in questionDict:
            questionDict[q] = []
            questionDict[q].append([u,usr_raw_dict[u][0],question_raw_dict[q][0], isAswer])
        else:
            questionDict[q].append([u,usr_raw_dict[u][0],question_raw_dict[q][0], isAswer])
    json.dump(userDict,fu)
    json.dump(questionDict,fq)

    fq.close()
    fu.close()


    print(invited[1])
    print(questions[1])
    print(users[1])


if __name__ == '__main__':
    # test()
    # question=json.load(open('question.json'))
    # user=json.load(open('user.json'))
    # print(len(question))
    # print(len(user))
    # i=1
    # for key in user:
    #     if len(user[key])==10:
    #         print(key,user[key])
    #         i+=1
    # print(i)
    with open('question_info.txt') as f:
        questions = [line.strip().split() for line in f]
        question2tag = {line[0]: line[1] for line in questions}
        json.dump(question2tag, open('question2tag.json','w'))
    print(question2tag)


