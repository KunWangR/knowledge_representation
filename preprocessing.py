#-*- coding:utf-8 -*-

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from itertools import groupby
import os
import pickle
from copy import deepcopy
import random

entity_nums=40943
relation_nums=18

#load entities from file
def load_entities(file_path):
    f=open(file_path)
    lines=f.readlines()
    entity_dict={}
    for i,line in enumerate(lines):
        if i==0:
            entity_nums=int(line)
        else:
            k_v=line.split('\t')
            entity_dict[int(k_v[0])]=int(k_v[1])
    return entity_nums,entity_dict

#load relations from file
def load_relations(file_path):
    f=open(file_path)
    lines=f.readlines()
    relation_dict={}
    for i,line in enumerate(lines):
        if i==0:
            relation_nums=int(line)
        else:
            k_v=line.split('\t')
            relation_dict[k_v[0]]=int(k_v[1])
    return relation_nums,relation_dict

#load train set from file
def load_train_set(train_file,entity_num):
    f=open(train_file)
    lines=f.readlines()
    train_set=[]
    p_h=[]
    p_t=[]
    p_r=[]
    for i,line in enumerate(lines):
        if i==0:
            train_num=int(line)
        else:
            (h,t,r)=line.split(' ')
            p_h.append(h)
            p_t.append(t)
            p_r.append(r)
    n_h=p_h
    n_r=p_r
    n_t=neg_t_generation_raw(p_t,entity_num)
    train_set.append(p_h)
    train_set.append(p_t)
    train_set.append(p_r)
    train_set.append(n_h)
    train_set.append(n_t)
    train_set.append(n_r)

    return train_num,train_set

#negtive sample generation with positive sample
def neg_sample_gengeration_raw(pos_sample,entity_num):
    pos_h=pos_sample[0]
    pos_t=pos_sample[1]
    pos_r=pos_sample[2]
    h_t_choice=np.random.uniform()
    if h_t_choice<0.5:
        neg_t=pos_sample[1]
        neg_r=pos_sample[2]
        while True:
            h=np.random.randint(entity_num)
            if h!=pos_h:
                break
        neg_h=h
    else:
        neg_h=pos_sample[0]
        neg_r=pos_sample[2]
        while True:
            t=np.random.randint(entity_num)
            if t!=pos_t:
                break
        neg_t=t
    sample={}
    sample['p_h']=pos_h
    sample['p_t']=pos_t
    sample['p_r']=pos_r
    sample['n_h']=neg_h
    sample['n_t']=neg_t
    sample['n_r']=neg_r
    return sample

def neg_sample_gengeration_filter(pos_sample,entity_num,tripleDict):
    pos_h=pos_sample[0]
    pos_t=pos_sample[1]
    pos_r=pos_sample[2]
    h_t_choice=np.random.uniform()
    if h_t_choice<0.5:
        neg_t=pos_sample[1]
        neg_r=pos_sample[2]
        while True:
            h=np.random.randint(entity_num)
            if (h,neg_t,neg_r) not in tripleDict:
                break
        neg_h=h
    else:
        neg_h=pos_sample[0]
        neg_r=pos_sample[2]
        while True:
            t=np.random.randint(entity_num)
            if (neg_h,t,neg_r) not in tripleDict:
                break
        neg_t=t
    sample={}
    sample['p_h']=pos_h
    sample['p_t']=pos_t
    sample['p_r']=pos_r
    sample['n_h']=neg_h
    sample['n_t']=neg_t
    sample['n_r']=neg_r
    return sample

#load test set from file
def load_test_set(test_file):
    df=pd.read_csv(test_file,sep=' ',encoding='utf-8',header=None)
    test_set={}
    test_set['p_h']=list(df[0])
    test_set['p_t']=list(df[1])
    test_set['p_r']=list(df[2])
    return test_set

class Trainingset(data.Dataset):
    def __init__(self,triple_file_path,train_file_path,filter=True):
        tripleTotal,tripleList,self.tripleDict=load_triples(triple_file_path)
        del tripleTotal,tripleList
        trainTripleTotal,self.TrainTripleList,trainTripleDict=load_triples(train_file_path)
        del trainTripleTotal,trainTripleDict
        self.filter=filter
    def __getitem__(self, index):
        pos_sample=self.TrainTripleList[index]
        if self.filter:
            sample=neg_sample_gengeration_filter(pos_sample,entity_nums,self.tripleDict)
        else:
            sample=neg_sample_gengeration_raw(pos_sample,entity_nums)
        return sample

    def __len__(self):
        return len(self.TrainTripleList)

class TestingSet(data.Dataset):
    def __init__(self,file_path):
        self.triples=pd.read_csv(file_path,sep=' ',encoding='utf-8',header=None)

    def __getitem__(self, index):
        pos_sample=self.triples.iloc[index,:]
        sample={}
        sample['h']=pos_sample[0]
        sample['t']=pos_sample[1]
        sample['r'] = pos_sample[2]
        return sample

    def __len__(self):
        return len(self.triples[0])

#load triple data set and return data_size, triple_tuple_list , triple_tuple_dict
def load_triples(file_path):
    with open(file_path,'r') as fr:
        i=0
        tripleList=[]
        for line in fr:
            if i==0:
                tripleTotal=int(line)
            else:
                line_split=line.split()
                head=int(line_split[0])
                tail=int(line_split[1])
                rel=int(line_split[2])
                tripleList.append((head,tail,rel))
            i+=1
    tripleDict={}
    for triple in tripleList:
        tripleDict[triple]=True
    return tripleTotal,tripleList,tripleDict

#write a triple list into a file, with three elements in form 【head tail rel】 per line
def process_list(tripleList,dataset,filename):
    with open(os.path.join('./data/', dataset, filename), 'w') as fw:
        fw.write(str(len(tripleList)) + '\n')
        for triple in tripleList:
            fw.write(str(triple[0]) + '\t' + str(triple[1]) + '\t' + str(triple[2]) + '\n')

#combine all dataset into one triple2id list and save to file
#In general, it is no use for the original knowledge database because it is just such a list
def tripleSet_combination(dataset):
    trainTotal,trainList,trainDict=load_triples('./data/'+dataset+'/train2id.txt')
    del trainTotal,trainDict
    testTotal,testList,testDict=load_triples('./data/'+dataset+'/test2id.txt')
    del testTotal,testDict
    validTotal, validList, validDict = load_triples('./data/' + dataset + '/valid2id.txt')
    del validTotal,validDict
    tripleList=trainList+testList+validList
    tripleList=list(set(tripleList))
    process_list(tripleList,dataset,'triple2id.txt')
    return tripleList


#calculate the statistic of datasets
def calculate_one_or_many(dataset_name):
    tripleTotal,tripleList,tripleDict=load_triples('./data/'+dataset_name+'/triple2id.txt')
    #sort the triplelist
    tripleList.sort(key=lambda x:(x[2],x[0],x[1]))
    grouped=[(k,list(g)) for k,g in groupby(tripleList,key=lambda x:x[2])]
    num_of_relations=len(grouped)
    head_per_tail_list=[0]*num_of_relations
    tail_per_head_list=[0]*num_of_relations

    one_to_one=[]
    one_to_many=[]
    many_to_one=[]
    many_to_many=[]

    for elem in grouped:
        headList=[]
        tailList=[]
        for triple in elem[1]:
            headList.append(triple[0])
            tailList.append(triple[1])
        headSet=set(headList)
        tailSet=set(tailList)
        head_per_tail=len(headList)/len(tailSet)
        tail_per_head=len(tailList)/len(headSet)
        head_per_tail_list[elem[0]]=head_per_tail
        tail_per_head_list[elem[0]]=tail_per_head

        if head_per_tail<1.5 and tail_per_head<1.5:
            one_to_one.append(elem[0])
        elif head_per_tail>=1.5 and tail_per_head<1.5:
            many_to_one.append(elem[0])
        elif head_per_tail<1.5 and tail_per_head>=1.5:
            one_to_many.append(elem[0])
        else:
            many_to_many.append(elem[0])

    #classify test triples according to the type of relation
    testTotal,testList,testDict=load_triples('./data/'+dataset_name+'/test2id.txt')
    testList.sort(key=lambda x:(x[2],x[0],x[1]))
    test_grouped=[(k,list(g)) for k,g in groupby(testList,key=lambda x:x[2])]

    one_to_one_list=[]
    one_to_many_list=[]
    many_to_one_list=[]
    many_to_many_list=[]

    for elem in test_grouped:
        if elem[0] in one_to_one:
            one_to_one_list+=elem[1]
        elif elem[0] in one_to_many:
            one_to_many_list+=elem[1]
        elif elem[0] in many_to_one:
            many_to_one_list+=elem[1]
        else:
            many_to_many_list+=elem[1]

    process_list(one_to_one_list, dataset_name, 'one_to_one_test.txt')
    process_list(one_to_many_list, dataset_name, 'one_to_many_test.txt')
    process_list(many_to_one_list, dataset_name, 'many_to_one_test.txt')
    process_list(many_to_many_list, dataset_name, 'many_to_many_test.txt')

    with open(os.path.join('./data/', dataset_name, 'head_tail_proportion.pkl'), 'wb') as fw:
        pickle.dump(tail_per_head_list, fw)
        pickle.dump(head_per_tail_list, fw)


def getThreeElements(tripleList):
	headList = [triple[0] for triple in tripleList]
	tailList = [triple[1] for triple in tripleList]
	relList = [triple[2] for triple in tripleList]
	return headList, tailList, relList

# Split the tripleList into #num_batches batches
def getBatchList(tripleList, num_batches):
	batchSize = len(tripleList) // num_batches
	batchList = [0] * num_batches
	for i in range(num_batches - 1):
		batchList[i] = tripleList[i * batchSize : (i + 1) * batchSize]
	batchList[num_batches - 1] = tripleList[(num_batches - 1) * batchSize : ]
	return batchList

#negtive tail entity generation without filter
def neg_t_generation_raw(triple,entityTotal):
    n_h=deepcopy(triple[0])
    n_r=deepcopy(triple[2])
    p_t=triple[1]
    while True:
        n_t=random.randrange(entityTotal)
        if n_t!=p_t:
            break
    newtriple=(n_h,n_t,n_r)
    return newtriple

#negtive head entity generation without filter
def neg_h_generation_raw(triple,entityTotal):
    n_t=deepcopy(triple[1])
    n_r=deepcopy(triple[2])
    p_h=triple[0]
    while True:
        n_h=random.randrange(entityTotal)
        if n_h!=p_h:
            break
    newtriple=(n_h,n_t,n_r)
    return newtriple
#get batch raw all
def getBatch_raw_all(tripleList,entityTotal):
    newTripleList=[neg_h_generation_raw(triple,entityTotal) if random.random()<0.5 else
                   neg_t_generation_raw(triple,entityTotal) for triple in tripleList]
    p_h,p_t,p_r=getThreeElements(tripleList)
    n_h,n_t,n_r=getThreeElements(newTripleList)
    sample={}
    sample['p_h']=p_h
    sample['p_t']=p_t
    sample['p_r']=p_r
    sample['n_h']=n_h
    sample['n_t']=n_t
    sample['n_r']=n_r
    return sample


#negtive head entity generation with filter
def neg_t_generation_filter(triple,entityTotal,tripleDict):
    n_h=triple[0]
    n_r=triple[2]
    while True:
        n_t=np.random.randint(entityTotal)
        if (n_h,n_t,n_r) not in tripleDict:
            break
    newTriple=(n_h,n_t,n_r)
    return newTriple

def neg_h_generation_filter(triple,entityTotal,tripleDict):
    n_t=triple[1]
    n_r=triple[2]
    while True:
        n_h=np.random.randint(entityTotal)
        if (n_h,n_t,n_r) not in tripleDict:
            break
    newTriple=(n_h,n_t,n_r)
    return newTriple

def getBatch_filter_all(tripleList,entityTotal,tripleDict):
    newTripleList=[neg_h_generation_filter(triple,entityTotal,tripleDict)
                   if np.random.uniform()<0.5 else
                   neg_t_generation_filter(triple,entityTotal,tripleDict)
                   for triple in tripleList]
    p_h,p_t,p_r=getThreeElements(tripleList)
    n_h,n_t,n_r=getThreeElements(newTripleList)
    samples={}
    samples['p_h'] = p_h
    samples['p_t'] = p_t
    samples['p_r'] = p_r
    samples['n_h'] = n_h
    samples['n_t'] = n_t
    samples['n_r'] = n_r
    return samples

def getBatch_raw_random(tripleList, batchSize, entityTotal):
    oldTripleList = random.sample(tripleList, batchSize)
    newTripleList = [neg_h_generation_raw(triple, entityTotal) if random.random() < 0.5
		else neg_t_generation_raw(triple, entityTotal) for triple in oldTripleList]
    p_h, p_t ,p_r = getThreeElements(oldTripleList)
    n_h, n_t, n_r = getThreeElements(newTripleList)
    samples={}
    samples['p_h'] = p_h
    samples['p_t'] = p_t
    samples['p_r'] = p_r
    samples['n_h'] = n_h
    samples['n_t'] = n_t
    samples['n_r'] = n_r
    return samples

def getBatch_filter_random(tripleList, batchSize, entityTotal, tripleDict):
    oldTripleList=random.sample(tripleList,batchSize)
    newTripleList=[neg_h_generation_filter(triple,entityTotal,tripleDict)
                   if np.random.uniform()<0.5 else
                   neg_t_generation_filter(triple,entityTotal,tripleDict)
                   for triple in tripleList]
    p_h,p_t,p_r=getThreeElements(oldTripleList)
    n_h,n_t,n_r=getThreeElements(newTripleList)
    samples={}
    samples['p_h'] = p_h
    samples['p_t'] = p_t
    samples['p_r'] = p_r
    samples['n_h'] = n_h
    samples['n_t'] = n_t
    samples['n_r'] = n_r
    return samples
