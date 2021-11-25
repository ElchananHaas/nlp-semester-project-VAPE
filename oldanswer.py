#!/usr/bin/python3 -W ignore::DeprecationWarning
# -*- coding:utf8 -*-
from collections import defaultdict
import sys
import re
import codecs
import re
import spacy
import os
from collections import defaultdict
nlp = spacy.load("en_core_web_lg")
from allennlp_models import pretrained
predictor = pretrained.load_predictor('structured-prediction-srl-bert')


#Implement word scaling?
class Document():
    def __init__(self,text):
        self.text=text
        self.doc=nlp(text)
        self.freq=defaultdict(int)
        for token in self.doc:
            self.freq[token.lemma_]+=1
    def answer(self,question):
        q=nlp(question)
        (best_sent,best_sentval)=(None,0)
        qsent=list(q.sents)[0]
        for asent in self.doc.sents:
            matchnum=self.phrase_sim(asent,qsent)
            if(matchnum>best_sentval):
                best_sentval=matchnum
                best_sent=asent
        asent=best_sent
        print("Q is: ",qsent)
        print("A is: ",asent)
        qtype=qsent[0].lemma_
        #print("type is: ",qtype)
        if qtype in ("do","be"):
            for token in asent:
                if token.lemma_ in ("not","except"):
                    return "no"
            return "yes"
        elif qtype in ("how","what","in","who","where"):
            verb_ans=qtype in ("how",) and qsent[1].pos_ not in ("ADJ","ADV")
            qslices=all_verbslices(qsent,True)
            aslices=all_verbslices(asent)
            print(qslices)
            print(aslices)
            bestans=(asent,asent)
            bestsc=-1
            if len(qslices)==0:
                return best_sent
            for ans in aslices:
                verbsc=0
                nounsc=0
                for qpart in qslices:
                    verbsim=ans[0].similarity(qpart[0])
                    nounsim=self.phrase_sim(ans[1],qpart[1])
                    sentsim=self.phrase_sim_ord(ans[1],qsent)
                    if(sentsim>.6):
                        nounsim=0 #Don't repeat question components
                    verbsc=max(verbsc,verbsim)
                    nounsc=max(nounsc,nounsim)
                print(ans,verbsim,nounsim)
                if verb_ans:
                    finalscore=nounsim-verbsim
                else:
                    finalscore=verbsim+nounsim
                if(finalscore>bestsc):
                    bestsc=finalscore
                    bestans=ans
            #print(bestans)
            if verb_ans:
                return str(bestans[0])
            else:
                return str(bestans[1])
            pass
        else:
            return best_sent
    def phrase_sim(self,p1,p2):
        return max(self.phrase_sim_ord(p1,p2),self.phrase_sim_ord(p2,p1))
    def phrase_sim_ord(self,p1,p2):
        score=0
        count=0
        for word in p1:
            if word.pos_ in ("PROPN","NUM","NOUN","VERB","ADJ","ADV"):
                ms=0
                weight=1
                if self.freq[word.lemma_]>5:
                    weight=5/self.freq[word.lemma_]
                for word2 in p2:
                    ms=max(ms,word.similarity(word2))
                score+=ms*weight
                count+=weight
        if count==0:
            return 0
        return score/count
def all_verbslices(nlpsent,addv=False):
    allen=predictor.predict_tokenized([token.text for token in nlpsent])
    #print(allen)
    parts=[]
    for qverb in allen['verbs']:
        slices=arg_slices(allen,qverb,nlpsent,addv)
        parts+=slices
    return parts
def arg_slices(sent,verb,nlpsent,addv):
    tag_types=["ARG0","ARG1","ARG2","ARGM-LOC"]
    v=tag_slice(sent,verb,"V",nlpsent)
    if addv:
        slices=[(v,v)]
    else:
        slices=[]
    for tag in tag_types:
        res=tag_slice(sent,verb,tag,nlpsent)
        if(len(res)>0):
            slices.append((v,res))
    return slices

def tag_slice(sent,verb,tag,nlpsent):
    words=[]
    start_i=None
    end_i=None
    for i in range(len(sent['words'])):
        if(verb['tags'][i] in ("B-"+tag,"I-"+tag)):
            if(start_i is None):
                start_i=i
            words.append(sent['words'][i])
        else:
            if(start_i is not None and end_i is None):
                end_i=i
    if end_i is None:
        end_i=len(sent['words'])
    if(start_i is None):
        return nlpsent[0:0]
    return nlpsent[start_i:end_i]
def common_verb(q,a):
    score=-1
    verbs=(None,None)
    for qverb in q['verbs']:
        for averb in a['verbs']:
            qw=nlp(qverb['verb'])
            aw=nlp(averb['verb'])
            sim=qw.similarity(aw)
            if(qverb['verb']=="is"):
                sim=sim-1/2
            if sim>score:
                score=sim
                verbs=(qverb,averb)
    return verbs
def phrase_score(phrase1,phrase2):
    child1=expand_children(phrase1)
    child2=expand_children(phrase2)
    #print(child1,child2)
    overlap=[]
    thresh=.95
    for word1 in child1:
        for word2 in child2:
            score=word1.similarity(word2)
            if(score>thresh or word1.lemma_==word2.lemma_):
                if(len(word1)<len(word2)):
                    overlap.append(word1)
                else:
                    overlap.append(word2)
    dedup_overlap=[]
    dedup_lemma=set(["be"])
    for word in overlap:
        if word.lemma_ not in dedup_lemma:
            dedup_lemma.add(word.lemma_)
            dedup_overlap.append(word)
    score=0
    for phrase in dedup_overlap:
        for word in phrase:
            if word.pos_ in ("PROPN","NUM"):
                score+=3
            elif word.pos_ in ("NOUN","VERB","ADJ"):
                score+=2
    if(score>4):
        pass
        #print(score,dedup_overlap)
    return score

def phrase_type(words,ptype):
    L=expand_children(words)
    L=[phrase for phrase in L if ptype in phrase._.labels]
    return L
def expand_children(words):
    return expand_children_old(words)

def expand_children_old(words):
    L=[words]
    for child in words._.children:
        L+=expand_children_old(child)
    return L 


def run(text,questions):
    answerer=Document(text)
    for line in questions:
        print(answerer.answer(line))

if __name__ == "__main__":
    input_file = sys.argv[1]
    question_file = sys.argv[2]
    with open(input_file,'r') as f:
        text=f.read()
    with open(question_file, 'r') as f:
        lines=list(f.readlines())
    run(text,lines)

#./answer ../Course-Project-Data/set2/a5.txt questions.txt

#./answer /home/elchanan/Documents/11411/52de896c-2232-11ec-84aa-0ed63bc4f033_Question_Answer_Dataset_v1.2/Question_Answer_Dataset_v1.2/S10/data/set1/a1.txt /home/elchanan/Documents/11411/52de896c-2232-11ec-84aa-0ed63bc4f033_Question_Answer_Dataset_v1.2/Question_Answer_Dataset_v1.2/S10/questions.txt