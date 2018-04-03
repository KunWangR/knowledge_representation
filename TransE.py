#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from knowledge_representation.preprocessing import *
from knowledge_representation.loss import marginLoss
entity_nums=40943
batch_nums=1415
relation_nums=18
hidden_size=20
margin=2
learning_rate=0.01
Epochs=20
Train_Data_path='./data/WN18/train2id.txt'
Test_Data_path='./data/WN18/test2id.csv'
Valid_Data_path='./data/WN18/valid2id.csv'

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


def train_transE(Train_Data_path):
    train_set=Trainingset(Train_Data_path)
    train_loader=data.DataLoader(train_set,batch_size=100,shuffle=True, num_workers=6)
    transE=TransE(ent_num=entity_nums,rel_num=relation_nums,hidden_size=hidden_size,margin=margin)
    optimizer=torch.optim.SGD(transE.parameters(),lr=learning_rate)
    #loss
    loss_func=torch.nn.MarginRankingLoss(margin,False)
    y = Variable(torch.Tensor([-1]))
    # last_mean_rank=1000000
    pbar=tqdm(total=Epochs)
    for i in range(Epochs):#epoch
        transE.norm_entity()
        epoch_loss = torch.FloatTensor([0.0])
        for j,batch_samples in enumerate(train_loader):#batch
            optimizer.zero_grad()  # 每次迭代清空上一次的梯度
            p_score,n_score=transE(batch_samples)
            loss=loss_func(p_score,n_score,y)
            #print('Epoch:', i, '|Step:', j, '|loss:', loss)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.data
            # print('loss:',loss,'epoch loss:',epoch_loss)
        print('epoch:', i, '|epoch loss:', epoch_loss.numpy())
        epoch_loss/=torch.FloatTensor([train_set.__len__()])
        pbar.update(1)
        # if (i+1)%50==0:
        #     mean_rank,hit10=test_evaluate('TransE',Test_Data_path)
        # print('Epoch:', i, '|loss:', epoch_loss,'|test mean rank:',mean_rank,'|hit_10:',hit10)
    pbar.close()
    torch.save(transE.state_dict(),'./models/TransE_params.pkl')# save only model params
    torch.save(transE,'./models/TransE.pkl')#save model and params

def train_transE_matual(Train_Data_path):
    tripleTotal, tripleList, tripleDict=load_triples(Train_Data_path)
    trainBatchList=getBatchList(tripleList,batch_nums)
    transE=TransE(ent_num=entity_nums,rel_num=relation_nums,hidden_size=hidden_size,margin=margin)
    # cretirion=nn.MarginRankingLoss(margin,False)
    # y=Variable(torch.Tensor([-1]))
    cretirion=marginLoss()
    optimizer=torch.optim.SGD(transE.parameters(),lr=learning_rate)
    pbar=tqdm(total=Epochs)
    for epoch in range(Epochs):
        transE.norm_entity()
        epoch_loss=torch.FloatTensor([0.0])
        random.shuffle(trainBatchList)
        for batchList in trainBatchList:
            sample_batch=getBatch_raw_all(batchList,entity_nums)
            transE.zero_grad()
            p_score,n_score=transE(sample_batch)
            # loss=cretirion(p_score,n_score,y)
            loss=cretirion.forward(p_score,n_score,margin)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.data
        epoch_loss/=torch.FloatTensor([tripleTotal])
        pbar.update(1)
        print('epoch:',epoch,'|loss:',epoch_loss.numpy())
    pbar.close()
    torch.save(transE.state_dict(),'./models/TransE_params.pkl')# save only model params
    torch.save(transE,'./models/TransE.pkl')#save model and params

def transE_evaluate(Test_Data_path):
    # transe=TransE(ent_num=entity_nums,rel_num=relation_nums,hidden_size=hidden_size,margin=margin)#init a model
    # transe.load_state_dict(torch.load('./models/transE_params.pkl'))#load model params
    model=torch.load('./models/TransE.pkl') #load model and params into model
    test_set=load_test_set(Test_Data_path)
    mean_rank=0
    hit_10=0
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

train_transE_matual(Train_Data_path)
transE_evaluate(Test_Data_path)
