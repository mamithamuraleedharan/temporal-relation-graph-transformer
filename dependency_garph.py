import spacy
nlp = spacy.load("en_core_web_sm")
sentence = nlp("John finished his work before he left")
print("Sentence : ",sentence)
for token in sentence:
    print(f"{token.text} : {token.pos_} : {token.dep_} : {token.head.text}")