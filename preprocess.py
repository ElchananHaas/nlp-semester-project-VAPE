import re
import spacy,benepar
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
from allennlp_models import pretrained
predictor = pretrained.load_predictor('structured-prediction-srl-bert')

class Document():
    def __init__(self,text):
        self.text=text
        self.doc=nlp(text)
    def answer(self,question):
        print()
        q=nlp(question)
        #print(dir(q))
        (best_sent,best_sentval)=(None,0)
        sent=list(q.sents)[0]
        for qsent in self.doc.sents:
            matchnum=phrase_score(qsent,sent)
            if(matchnum>best_sentval):
                #print(matchnum)
                best_sentval=matchnum
                best_sent=qsent
        q=predictor.predict(question)
        ans=predictor.predict(str(best_sent))
        print(q)
        print(ans)
        #print(sent._.parse_string)


def noun_strategy(q,sent):
    q_lemma=set([word.lemma_ for word in q])
    for i,word in enumerate(sent):
        if(word.ent_iob==3 and (word.lemma_ not in q_lemma)):
            end=i+1
            while(end<len(sent) and sent[end].ent_iob==1):
                end=end+1
            np=sent[i:end]
            answer=str(np)+" "+str(q[1:])
            return answer

def verb_strategy(q,sent,verb,noun):
    bestscore=-1
    for i in range(len(sent)-len(verb)+1):
        score=verb.similarity(sent[i:i+len(verb)])
        if(score>bestscore):
            bestscore=score
            vp=containing_type(sent[i],"VP")
    if vp is not None:
        connector=" "
        verb_list=("be","can")
        if(vp[0].lemma_ not in verb_list):
            pass 
            for word in q:
                #print(word.lemma_)
                if word.lemma_ in verb_list:
                    connector=" "+str(word)+" "  
                    break   
        answer=str(noun)+connector+str(vp)
        return answer
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
    if(len(dedup_overlap)>0):
        pass
        #print(dedup_overlap)
    score=sum([len(phrase) for phrase in dedup_overlap])
    return score
def expand_children(words):
    L=[words]
    for child in words._.children:
        L+=expand_children(child)
    return L                  
        #print(list(sent._.children))
        #print([chunk.text for chunk in self.doc.noun_chunks])

def containing_type(phrase,type):
    if phrase is None:
        return None
    #print(phrase,phrase._.labels)
    if(type in phrase._.labels):
        return phrase 
    return containing_type(phrase._.parent,type)




def load_file(path):
    with open(path,'r') as f:
        text=f.read()
        return Document(text)

answerer=load_file("../Course-Project-Data/set2/a5.txt")
#answerer.answer("What is Delta Cancri also known as?")
#answerer.answer("What is cancer bordered by?")
#answerer.answer("What is the brightest star in Cancer?")
answerer.answer("What is cancers astrological symbol?")
#answerer.answer("What latitudes can cancer be seen at?")
#answerer.answer("What open cluster is located right in the centre of cancer?")
#answerer.answer("What month was cancer associated with?")
#answerer.answer("Who followed cancer in ancient Greece?")
#answerer.answer("Who placed the crab among the stars?")
#answerer.answer("Who gave Castor immortality and why?")


"""
s1="Cancer is a medium-sized constellation that is bordered by Gemini to the west, Lynx to the north, Leo Minor to the northeast, Leo to the east, Hydra to the south, and Canis Minor to the southwest."
print(list(nlp(s1).sents)[0]._.parse_string)
s2="bordered by" """