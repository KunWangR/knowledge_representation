#-*- coding:utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from tqdm import tqdm as tqdm
import time
from knowledge_representation.preprocessing import Trainingset,load_test_set


entity_nums=40943
relation_nums=18
hidden_size=20
margin=2
learning_rate=0.01
Epochs=1
train_batch_size=100
Train_Data_path='./data/WN18/train2id.csv'
Test_Data_path='./data/WN18/test2id.csv'
Valid_Data_path='./data/WN18/valid2id.csv'


class TransH(nn.Module):
    def __init__(self,ent_num,rel_num,hidden_size,margin):
        super(TransH,self).__init__()
        self.ent_num=ent_num
        self.rel_num=rel_num
        self.hidden_size=hidden_size
        self.margin=margin
        self.ent_embedding=nn.Embedding(self.ent_num,self.hidden_size)
        self.rel_embedding=nn.Embedding(self.rel_num,self.hidden_size)
        self.norm_vector=nn.Embedding(self.rel_num,self.hidden_size)
        self.init_weights()
        self.ent_embedding.weight.data=F.normalize(self.ent_embedding.weight.data,p=2,dim=1)
        self.rel_embedding.weight.data=F.normalize(self.rel_embedding.weight.data,p=2,dim=1)
        self.norm_vector.weight.data=F.normalize(self.norm_vector.weight.data,p=2,dim=1)

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_embedding.weight.data)
        nn.init.xavier_uniform(self.rel_embedding.weight.data)
        nn.init.xavier_uniform(self.norm_vector.weight.data)

    def _transfer(self,e,norm):
        return e-torch.sum(e*norm,1,True)*norm

    def _calc(self,h,t,r):
        return torch.abs(h+r-t)

    def loss_func(self,p_score,n_score):
        criterion=nn.MarginRankingLoss(self.margin,False)
        y=Variable(torch.Tensor([-1]))
        loss=criterion(p_score,n_score,y)
        return loss

    def forward(self,samples):
        p_h_e=self.ent_embedding(Variable(samples['p_h']))
        p_t_e=self.ent_embedding(Variable(samples['p_t']))
        p_r_e=self.rel_embedding(Variable(samples['p_r']))

        n_h_e=self.ent_embedding(Variable(samples['n_h']))
        n_t_e=self.ent_embedding(Variable(samples['n_t']))
        n_r_e=self.rel_embedding(Variable(samples['n_r']))

        p_norm=self.norm_vector(Variable(samples['p_r']))
        n_norm = self.norm_vector(Variable(samples['n_r']))

        p_h=self._transfer(p_h_e,p_norm)
        p_t=self._transfer(p_t_e,p_norm)
        p_r=p_r_e
        n_h=self._transfer(n_h_e,n_norm)
        n_t=self._transfer(n_t_e,n_norm)
        n_r=n_r_e

        _p_score=self._calc(p_h,p_t,p_r)
        _n_score=self._calc(n_h,n_t,n_r)

        p_score=torch.sum(_p_score,1)
        n_score=torch.sum(_n_score,1)

        loss=self.loss_func(p_score,n_score)
        return loss

    def predict(self,predict_h,predict_t,predict_r):
        p_h_e=self.ent_embedding(Variable(torch.LongTensor(predict_h)))
        p_t_e=self.ent_embedding(Variable(torch.LongTensor(predict_t)))
        p_r_e=self.rel_embedding(Variable(torch.LongTensor(predict_r)))

        p_norm=self.norm_vector(Variable(torch.LongTensor(predict_r)))

        p_h=self._transfer(p_h_e,p_norm)
        p_t=self._transfer(p_t_e,p_norm)
        p_r=p_r_e
        _p_score=self._calc(p_h,p_t,p_r)
        p_score=torch.sum(_p_score,1)
        return p_score


def train_TransH(Train_Data_path):
    train_set=Trainingset(Train_Data_path)
    train_loader=data.DataLoader(train_set,batch_size=train_batch_size,shuffle=True, num_workers=6)
    transH=TransH(ent_num=entity_nums,rel_num=relation_nums,hidden_size=hidden_size,margin=margin)
    optimizer=torch.optim.SGD(transH.parameters(),lr=learning_rate)
    #loss
    loss_func=torch.nn.MarginRankingLoss(margin,False)
    y = Variable(torch.Tensor([-1]))
    # last_mean_rank=1000000
    pbar=tqdm(total=Epochs)
    for i in range(Epochs):#epoch
        pbar.update(1)

        epoch_loss = 0.0
        for j,batch_samples in enumerate(train_loader):#batch
            optimizer.zero_grad()  # 每次迭代清空上一次的梯度
            p_score,n_score=transH(batch_samples)
            loss=loss_func(p_score,n_score,y)
            #print('Epoch:', i, '|Step:', j, '|loss:', loss)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.data
            # print('loss:',loss,'epoch loss:',epoch_loss)
        print('epoch:', i, '|epoch loss:', epoch_loss.data.numpy())
        epoch_loss/=train_set.__len__()
        # if (i+1)%50==0:
        #     mean_rank,hit10=test_evaluate('TransE',Test_Data_path)
        # print('Epoch:', i, '|loss:', epoch_loss,'|test mean rank:',mean_rank,'|hit_10:',hit10)
    pbar.close()
    file_name = '_'.join(['l',str(learning_rate),
                          'bat',str(train_batch_size),
                          'm',str(margin),
                          'op','SGD',
                          'em',str(hidden_size),
                          't',str(int(time.time()))
                          ])
    torch.save(transH.state_dict(),'./models/transH_params.pkl')# save only model params
    torch.save(transH,'./models/transH.pkl')#save model and params


def transH_evaluate(Test_Data_path):
    # transe=TransH(ent_num=entity_nums,rel_num=relation_nums,hidden_size=hidden_size,margin=margin)#init a model
    # transe.load_state_dict(torch.load('./models/transH_params.pkl'))#load model params
    model=torch.load('./models/transH.pkl') #load model and params into model
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