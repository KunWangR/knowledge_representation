#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
import time
from sklearn.metrics import pairwise_distances
from knowledge_representation.preprocessing import *
from knowledge_representation.loss import marginLoss
entity_nums=40943
batch_nums=1415
relation_nums=18
hidden_size=20
margin=2
learning_rate=0.01
Epochs=1000
Filter=False
Early_Stopping_Round=30
Triple_Data_path='./data/WN18/triple2id.txt'
Train_Data_path='./data/WN18/train2id.txt'
train_data_path='./data/WN18/train2id.csv'
Test_Data_path='./data/WN18/test2id.txt'
test_Data_path='./data/WN18/test2id.csv'
Valid_Data_path='./data/WN18/valid2id.txt'
valid_Data_path='./data/WN18/valid2id.csv'

class TransE(nn.Module):
    def __init__(self,ent_num,rel_num,hidden_size,margin):
        super(TransE,self).__init__()
        self.ent_num=ent_num
        self.rel_num=rel_num
        self.margin=margin
        self.hidden_size=hidden_size
        self.ent_embedding = nn.Embedding(self.ent_num, self.hidden_size)
        self.rel_embedding = nn.Embedding(self.rel_num, self.hidden_size)

        self.init_weights()
        #l2 norm the relation
        self.rel_embedding.weight.data=F.normalize(self.rel_embedding.weight.data,p=2,dim=1)

    def norm_entity(self):
        self.ent_embedding.weight.data=F.normalize(self.ent_embedding.weight.data,p=2,dim=1)

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_embedding.weight.data)
        nn.init.xavier_uniform(self.rel_embedding.weight.data)

    def _calc(self,h,t,r):
        return torch.abs(h+r-t)

    def loss_func(self,p_score,n_score):
        criterion=nn.MarginRankingLoss(self.margin,False)
        y=Variable(torch.Tensor([-1]))
        loss=criterion(p_score,n_score,y)
        return loss

    def forward(self,samples):
        #pos_h, pos_t, pos_r, neg_h, neg_t, neg_r
        p_h=self.ent_embedding(Variable(torch.LongTensor(samples['p_h'])))
        p_t=self.ent_embedding(Variable(torch.LongTensor(samples['p_t'])))
        p_r=self.rel_embedding(Variable(torch.LongTensor(samples['p_r'])))

        n_h=self.ent_embedding(Variable(torch.LongTensor(samples['n_h'])))
        n_t=self.ent_embedding(Variable(torch.LongTensor(samples['n_t'])))
        n_r=self.rel_embedding(Variable(torch.LongTensor(samples['n_r'])))

        _p_score=self._calc(p_h,p_t,p_r)
        _n_score=self._calc(n_h,n_t,n_r)

        p_score=torch.sum(_p_score,1)
        n_score=torch.sum(_n_score,1)

        # loss=self.loss_func(p_score,n_score)

        return p_score,n_score

    def predict(self,predict_h,predict_t,predict_r):
        # print(type(predict_h),predict_h[0],type(predict_h[0]))
        # print(Variable(torch.LongTensor(predict_h)))
        p_h=self.ent_embedding(Variable(torch.LongTensor(predict_h)))
        p_t = self.ent_embedding(Variable(torch.LongTensor(predict_t)))
        p_r = self.rel_embedding(Variable(torch.LongTensor(predict_r)))

        _p_score=self._calc(p_h,p_t,p_r)
        p_score=torch.sum(_p_score,1)
        #print(p_score)
        return p_score


def train_transE(Triple_Data_path,Train_Data_path,usefilter):
    train_set=Trainingset(Triple_Data_path,Train_Data_path,usefilter)
    train_loader=data.DataLoader(train_set,batch_size=50,shuffle=True, num_workers=6)
    transE=TransE(ent_num=entity_nums,rel_num=relation_nums,hidden_size=hidden_size,margin=margin)
    optimizer=optim.SGD(transE.parameters(),lr=learning_rate)
    #loss
    # loss_func=torch.nn.MarginRankingLoss(margin,False)
    # y = Variable(torch.Tensor([-1]))
    cretirion=marginLoss()
    # last_mean_rank=1000000
    pbar=tqdm(total=Epochs)
    for i in range(Epochs):#epoch
        transE.norm_entity()
        epoch_loss = torch.FloatTensor([0.0])
        for j,batch_samples in enumerate(train_loader):#batch
            #print(batch_samples)
            optimizer.zero_grad()  # 每次迭代清空上一次的梯度
            p_score,n_score=transE.forward(batch_samples)
            #loss=loss_func(p_score,n_score,y)
            loss = cretirion.forward(p_score, n_score, margin)
            #print('Epoch:', i, '|Step:', j, '|loss:', loss)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.data
            # print('loss:',loss,'epoch loss:',epoch_loss)
        epoch_loss/=train_set.__len__()
        print('epoch:', i, '|epoch loss:', epoch_loss.numpy())
        pbar.update(1)
        # if (i+1)%50==0:
        #     mean_rank,hit10=test_evaluate('TransE',Test_Data_path)
        # print('Epoch:', i, '|loss:', epoch_loss,'|test mean rank:',mean_rank,'|hit_10:',hit10)
    pbar.close()
    torch.save(transE.state_dict(),'./models/TransE_params.pkl')# save only model params
    torch.save(transE,'./models/TransE.pkl')#save model and params

def train_transE_matual(Triple_Data_path,Train_Data_path,Valid_Data_path,Test_Data_path,usefilter):
    tripleTotal,tripleList,tripleDict=load_triples(Triple_Data_path)

    validTripleTotal,validTripleList,validTripleDict=load_triples(Valid_Data_path)

    trainTripleTotal,trainTripleList,trainTripleDict =load_triples(Train_Data_path)

    testTripleTotal, testTripleList, testTripleDict = load_triples(Test_Data_path)

    trainBatchList=getBatchList(trainTripleList,batch_nums)
    transE=TransE(ent_num=entity_nums,rel_num=relation_nums,hidden_size=hidden_size,margin=margin)
    # cretirion=nn.MarginRankingLoss(margin,False)
    # y=Variable(torch.Tensor([-1]))
    cretirion=marginLoss()
    optimizer=torch.optim.SGD(transE.parameters(),lr=learning_rate)
    pbar=tqdm(total=300)
    early_stopping_round=0
    for epoch in range(499,Epochs):
        transE.norm_entity()
        epoch_loss=torch.FloatTensor([0.0])
        random.shuffle(trainBatchList)
        for batchList in trainBatchList:
            if usefilter:
                sample_batch=getBatch_filter_all(batchList,entity_nums,tripleDict)
            else:
                sample_batch=getBatch_raw_all(batchList,entity_nums)
            transE.zero_grad()
            p_score,n_score=transE(sample_batch)
            # loss=cretirion(p_score,n_score,y)
            loss=cretirion.forward(p_score,n_score,margin)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.data
        epoch_loss/=torch.FloatTensor([trainTripleTotal])
        pbar.update(1)
        if epoch%10==0:
            print('epoch:',epoch,'|loss:',epoch_loss.numpy())
            if usefilter==True:
                batch_samples=getBatch_filter_random(validTripleList,len(batchList),entity_nums,tripleDict)
            else:
                batch_samples=getBatch_raw_random(validTripleList, len(batchList), entity_nums)
            pos_valid,neg_valid=transE(batch_samples)
            #print(pos_valid,margin)
            losses=cretirion(pos_valid,neg_valid,margin)
            print('Epoch:',epoch,'|Valid batch loss:',losses.data.numpy())
        if epoch>=499:
            ent_embeddings=transE.ent_embedding.weight.data.numpy()
            rel_embeddings=transE.rel_embedding.weight.data.numpy()
            if epoch==499:
                best_meanrank,hit_10=transE_evaluate_allin(validTripleList, ent_embeddings, rel_embeddings)

            else:
                now_meanrank,hit_10=transE_evaluate_allin(validTripleList, ent_embeddings, rel_embeddings)
                if now_meanrank<best_meanrank:
                    best_meanrank=now_meanrank
                    test_meanrank,test_hit_10=transE_evaluate_allin(testTripleList,ent_embeddings,rel_embeddings)
                    print('Epoch:',epoch,'|test mean rank:',test_meanrank,'|test hit@10:',test_hit_10)
                else:
                    early_stopping_round+=1
            if early_stopping_round>=Early_Stopping_Round:
                break
    pbar.close()
    torch.save(transE.state_dict(),'./models/TransE_params.pkl')# save only model params
    torch.save(transE,'./models/TransE.pkl')#save model and params


def transE_evaluate_1by1_helper(Test_Data_path):
    model=TransE(ent_num=entity_nums,rel_num=relation_nums,hidden_size=hidden_size,margin=margin)#init a model
    model.load_state_dict(torch.load('./models/TransE_params.pkl'))#load model params
    # model=torch.load('./models/TransE.pkl') #load model and params into model
    ent_embeddings=model.ent_embedding.weight.data.numpy()
    rel_embeddings=model.rel_embedding.weight.data.numpy()
    testTotal,testList,testDict=load_triples(Test_Data_path)
    return testList,model,ent_embeddings,rel_embeddings
#evaluate transE on testset one by one
def transE_evaluate_1by1_pred(testList,model):
    start_time=time.time()
    mean_rank=0
    hit_10=0
    test_set={}
    test_set['p_h']=[triple[0] for triple in testList]
    test_set['p_t']=[triple[1] for triple in testList]
    test_set['p_r']=[triple[2] for triple in testList]

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
    end_time=time.time()
    print('total time:',end_time-start_time)
    return mean_rank,hit_10

def transE_evaluate_1by1_emb(testList,ent_embeddiings,rel_embeddings):
    start_time=time.time()
    mean_rank=0
    hit_10=0
    test_set={}
    test_set['p_h']=[triple[0] for triple in testList]
    test_set['p_t']=[triple[1] for triple in testList]
    test_set['p_r']=[triple[2] for triple in testList]

    for i,(h,t,r) in enumerate(zip(test_set['p_h'],test_set['p_t'],test_set['p_r'])):
        b_h=ent_embeddiings[h]
        b_r=rel_embeddings[r]
        b_t_e=b_h+b_r
        b_score=pairwise_distances(b_t_e.reshape(1, -1),ent_embeddiings,metric='manhattan',n_jobs=6)
        # print('b_score:',b_score[0])
        b_ranks=np.argsort(b_score[0])
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
    end_time=time.time()
    print('total time:',end_time-start_time)
    return mean_rank,hit_10

def transE_evaluate_allin_helper(Test_Data_path):
    transE = TransE(ent_num=entity_nums, rel_num=relation_nums, hidden_size=hidden_size, margin=margin)  # init a model
    transE.load_state_dict(torch.load('./models/TransE_params.pkl'))  # load model params
    # transE=torch.load('./models/transE.pkl')
    ent_embeddings = transE.ent_embedding.weight.data.numpy()
    rel_embeddings = transE.rel_embedding.weight.data.numpy()
    testTotal, testList, testDict = load_triples(Test_Data_path)
    return testList,ent_embeddings,rel_embeddings

#evaluate transE with testset sample all in
def transE_evaluate_allin(testList,ent_embeddings,rel_embeddings):
    start=time.time()

    headList=[triple[0] for triple in testList]
    tailList=[triple[1] for triple in testList]
    relList=[triple[2] for triple in testList]

    h_e=ent_embeddings[headList]
    t_e=ent_embeddings[tailList]
    r_e=rel_embeddings[relList]

    c_t_e=h_e+r_e
    dist=pairwise_distances(c_t_e,ent_embeddings,metric='manhattan',n_jobs=6)
    rankArrayTail=np.argsort(dist,axis=1)

    rankListTail=[int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList,rankArrayTail)]

    isHit10ListTail=[x for x in rankListTail if x<10]
    totalRank=sum(rankListTail)
    hit10Count=len(isHit10ListTail)
    tripleCount=len(rankListTail)

    mean_rank=totalRank/tripleCount
    hit_10=hit10Count/tripleCount
    print('Test data --> mean rank:',mean_rank,'hit10:',hit_10)
    end=time.time()
    print('all in total time:',end-start)
    return mean_rank,hit_10
# train_transE_matual(Train_Data_path)

# #evaluate transE one by one with model prediction
# testList,model,ent_embeddings,rel_embeddings=transE_evaluate_1by1_helper(Test_Data_path)
# transE_evaluate_1by1_pred(testList,model)

##evaluate transE one by one with embedding
# transE_evaluate_1by1_emb(testList,ent_embeddings,rel_embeddings)

train_transE(Triple_Data_path,Train_Data_path,Filter)
#train_transE_matual(Triple_Data_path,Train_Data_path,Valid_Data_path,Test_Data_path,Filter)
testList,ent_embeddings,rel_embeddings=transE_evaluate_allin_helper(Test_Data_path)
transE_evaluate_allin(testList,ent_embeddings,rel_embeddings)
