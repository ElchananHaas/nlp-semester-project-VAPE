import spacy,benepar
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
nlpq=nlp("The only places lacking indigenous ants are Antarctica and certain remote or inhospitable islands.")
for word in nlpq:
    print(word.pos_)