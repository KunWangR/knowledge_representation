#-*- coding:utf-8 -*-

from knowledge_representation.preprocessing import *
from knowledge_representation.TransE import TransE
from knowledge_representation.TransH import TransH
from torch.autograd import Variable

import torch
import torch.utils.data as data
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances
import multiprocessing
import time
import math


entity_nums=40943
relation_nums=18
hidden_size=50
margin=2
learning_rate=0.01
Epochs=10
Train_Data_path='./data/WN18/train2id.txt'
Test_Data_path='./data/WN18/test2id.txt'
Valid_Data_path='./data/WN18/valid2id.txt'


def model_init(model_name):
    if model_name=='TransE':
        transE = TransE(ent_num=entity_nums, rel_num=relation_nums, hidden_size=hidden_size, margin=margin)
        return transE
    elif model_name=='TransH':
        transH=TransH(ent_num=entity_nums,rel_num=relation_nums,hidden_size=hidden_size,margin=margin)
        return transH
    else:
        print('There is no such model.')
        return None

#model run with model_name as input
def model_run(model_name):
    train_set=Trainingset(Train_Data_path)
    train_loader=data.DataLoader(train_set,batch_size=100,shuffle=True, num_workers=6)
    model=model_init(model_name)
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
    #loss
    loss_func=torch.nn.MarginRankingLoss(margin,False)
    y = Variable(torch.Tensor([-1]))
    # last_mean_rank=1000000
    pbar=tqdm(total=Epochs)
    for i in range(Epochs):#epoch
        pbar.update(1)
        model.norm_entity()
        epoch_loss = torch.FloatTensor([0.0])
        for j,batch_samples in enumerate(train_loader):#batch
            optimizer.zero_grad()  # 每次迭代清空上一次的梯度
            p_score,n_score=model.forward(batch_samples)
            loss=loss_func(p_score,n_score,y)
            #print('Epoch:', i, '|Step:', j, '|loss:', loss)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.data

            # print('loss:',loss.data.numpy(),'epoch loss:',epoch_loss.data.numpy())
        epoch_loss/=train_set.__len__()
        print('Epoch:', i, 'epoch loss:', epoch_loss.numpy())
        # if (i+1)%50==0:
        #     mean_rank,hit10=test_evaluate('TransE',Test_Data_path)
        # print('Epoch:', i, '|loss:', epoch_loss,'|test mean rank:',mean_rank,'|hit_10:',hit10)
    pbar.close()
        # mean_rank,hit_10=test_evaluate(Valid_Data_path)
        # if last_mean_rank<mean_rank:  # how to early stopping? maybe mean_rank no change in 5(10) epoch
        #     break
        # else:
        #     last_mean_rank=mean_rank

    torch.save(model.state_dict(),'./models/'+model_name+'_params.pkl')# save only model params
    torch.save(model,'./models/'+model_name+'.pkl')#save model and params


def test_evaluate_helper(model_name,TestData_path):
    if model_name=='TransE':
        model=TransE(ent_num=entity_nums,rel_num=relation_nums,hidden_size=hidden_size,margin=margin)#init a model
        model.load_state_dict(torch.load('./models/TransE_params.pkl'))#load model params
    elif model_name=='TransH':
        model=TransH(ent_num=entity_nums,rel_num=relation_nums,hidden_size=hidden_size,margin=margin)#init a model
        model.load_state_dict(torch.load('./models/TransH_params.pkl'))#load model params
    # model=torch.load('./models/'+model_name+'.pkl') #load model and params into model
    tripleTotal,tripleList,tripleDict=load_triples(TestData_path)
    return tripleList,model


#model evaluate on test set
def test_evaluate(tripleList,model):
    mean_rank=0
    hit_10=0
    test_set={}
    test_set['p_h']=[triple[0] for triple in tripleList]
    test_set['p_t']=[triple[1] for triple in tripleList]
    test_set['p_r']=[triple[2] for triple in tripleList]

    for i,(h,t,r) in enumerate(zip(test_set['p_h'],test_set['p_t'],test_set['p_r'])):
        b_h=[int(h)]*entity_nums
        b_r=[int(r)]*entity_nums
        b_t=[int(i) for i in range(entity_nums)]
        b_score=model.predict(b_h,b_t,b_r)
        # print('b_score:',b_score)
        b_ranks=np.argsort(b_score.data.numpy())
        # print('b_ranks:',b_ranks)
        rank_t=int(np.argwhere(b_ranks==t))#
        #print(i,'sample','|head:',h,'|tail:',t,'|r:',r,'|rank:',rank_t)
        #print('|rank:',rank_t)
        mean_rank+=rank_t+1
        if rank_t<10:
            hit_10+=1
    mean_rank/=len(test_set['p_h'])
    hit_10/=len(test_set['p_h'])
    print('mean_rank:',mean_rank,'|hit@10:',hit_10)
    return mean_rank,hit_10

# model_run('TransE')
# test_evaluate('TransE',Test_Data_path)


#------------multiple progress for evaluation-------------------------------------------------
def evaluation_TransH(model_path,testList,tripleDict,head=0):
    transH=torch.load(model_path)
    # testList.sort(key=lambda x:(x[2],x[0],x[1]))
    # grouped=[(k,list(g)) for k,g in groupby(testList,key=lambda x:x[2])]

    ent_embeddings=transH.ent_embedding.weight.data
    rel_embeddings=transH.rel_embedding.weight.data
    norm_embeddings=transH.norm_vector.weight.data

    # one time sppedup matirx calculation
    head_list=[triple[0] for triple in testList]
    tail_list=[triple[1] for triple in testList]
    rel_list=[triple[2] for triple in testList]

    h_e=ent_embeddings[head_list]
    t_e=ent_embeddings[tail_list]
    r_e=rel_embeddings[rel_list]


#------------------multiprocess for transE evaluation---------------------------------------------
def evaluation_transE_helper(testList,tripleDict,ent_embeddings,rel_embeddings,head=0):
    # one time sppedup matirx calculation
    head_list=[triple[0] for triple in testList]
    tail_list=[triple[1] for triple in testList]
    rel_list=[triple[2] for triple in testList]

    h_e=ent_embeddings[head_list]
    t_e=ent_embeddings[tail_list]
    r_e=rel_embeddings[rel_list]

    #evaluate the prediction of only tail entity
    c_t_e=h_e+r_e
    dist=pairwise_distances(c_t_e,ent_embeddings,metric='manhattan')# default is euclidean
    rankArrayTail=np.argsort(dist,axis=1)
    rankListTail=[int(np.argwhere(elem[1]==elem[0])) for elem in zip(tail_list,rankArrayTail)]
    isHit10ListTail=[x for x in rankListTail if x<10]
    totalRank=sum(rankListTail)
    hit10Count=len(isHit10ListTail)
    tripleCount=len(rankListTail)

    return hit10Count,totalRank,tripleCount

class MyProcessTransE(multiprocessing.Process):
    def __init__(self,L,tripleDict,ent_embeddings,rel_embeddings,queue=None):
        super(MyProcessTransE,self).__init__()
        self.L=L
        self.queue=queue
        self.tripleDict=tripleDict
        self.ent_embeddings=ent_embeddings
        self.rel_embeddings=rel_embeddings

    def run(self):
        while True:
            testList=self.queue.get()
            try:
                self.process_data(testList,self.tripleDict,self.ent_embeddings,self.rel_embeddings,
                                  self.L)
            except:
                time.sleep(5)
                self.process_data(testList,self.tripleDict,self.ent_embeddings,self.rel_embeddings,
                                  self.L)
            self.queue.task_done()

    def process_data(self,testList,tripleDict,ent_embeddings,rel_embeddings,L):
        hit10Count,totalRank,tripleCount=evaluation_transE_helper(testList,tripleDict,ent_embeddings,rel_embeddings)
        L.append((hit10Count,totalRank,tripleCount))

def evaluation_transE(testList,tripleDict,ent_embeddings,rel_embeddings,num_processes=multiprocessing.cpu_count()):
    start_time=time.time()
    #split the testList into #num_processes parts
    len_split=math.ceil(len(testList)/num_processes)
    testListSplit=[testList[i:i+len_split] for i in range(0,len(testList),len_split)]
    with multiprocessing.Manager() as manager:
        #create a public writable list to store the result
        L=manager.list()
        queue=multiprocessing.JoinableQueue()
        workerList=[]
        for i in range(num_processes):
            worker=MyProcessTransE(L,tripleDict,ent_embeddings,rel_embeddings,queue=queue)
            workerList.append(worker)
            worker.daemon=True
            worker.start()

        for subList in testListSplit:
            queue.put(subList)

        queue.join()
        resultList=list(L)
        #Terminate the worker after execution, to avoid memory leaking
        for worker in workerList:
            worker.terminate()

    hit10=sum([elem[0] for elem in resultList])/len(testList)
    meanrank=sum(elem[1] for elem in resultList)/len(testList)
    print('Meanrank: %.6f'%meanrank)
    print('Hit@10: %.6f' %hit10)
    end_time=time.time()
    print('total time:',end_time-start_time)

    return hit10,meanrank
#---------------testing for multi progress-----------------------------------------------
# testTotal,testTripleList,testTripleDict=load_triples(Test_Data_path)
# transE=torch.load('./models/transE.pkl')
# ent_embeddings=transE.ent_embedding.weight.data.numpy()
# rel_embeddings=transE.rel_embedding.weight.data.numpy()
# evaluation_transE(testList=testTripleList,tripleDict=testTripleDict,
#                   ent_embeddings=ent_embeddings,rel_embeddings=rel_embeddings
#                   )


#----------------testing for overwriting data loader-----------------------------------------------
