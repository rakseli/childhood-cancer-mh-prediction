import sys
import re
from pathlib import Path
home=str(Path.home())
sys.path.append(f'{home}/Turku-neural-parser-pipeline/')
from tnparser.pipeline import read_pipelines, Pipeline
from nltk.corpus import stopwords
from collections import Counter

model_path = f'{home}/Turku-neural-parser-pipeline/'
available_pipelines=read_pipelines(f'{model_path}models_fi_tdt_dia/pipelines.yaml') 
#instantiate parse_plaintext pipeline
p=Pipeline(available_pipelines["parse_plaintext"])

#regex
re_text = re.compile('[^a-zäö ]')
re_space = re.compile('\s{2,}')
re_tag_1 = re.compile('<br><br>')
re_tag_2 = re.compile('<br>')
tags = [re_tag_1,re_tag_2]
#stopwords
stop_words = list(set(stopwords.words('finnish')))
stop_words.remove('ei')
measures = ['cm','kg','ml','mg','mmolmol','mmoll','vrk','vko']
stop_words.extend(measures)
stopwords_dict = Counter(stop_words)

def remove_stopwords(str_doc):
    #lowercase
    str_doc = str_doc.lower()
    #remove everything else than a-zäö\s
    str_doc = re.sub(re_text,"",str_doc)
    str_doc = ' '.join(word for word in str_doc.split() if word not in stopwords_dict)
    #remove extra \s
    str_doc = re.sub(re_space," ",str_doc)
    return str_doc

def remove_tags(str_doc):
    #remove tags
    for t in tags:
        str_doc = re.sub(t,'',str_doc)
    #remove extra \s
    str_doc = re.sub(re_space," ",str_doc)
    return str_doc
    
def extract_lemmatized(parser_output):
    lines = parser_output.splitlines()
    text = []
    for l in lines:
        if l.startswith('#') or l.startswith('\n') or l=='':
            continue
        else:
            splitted = l.split('\t')
            lemma = splitted[2]
            text.append(lemma)
            
    lemmatized_str = " ".join(text)
    return lemmatized_str


def lemmatize(text):
    text = remove_tags(text)
    parsed = p.parse(text)
    lemmatized_text = extract_lemmatized(parsed)
    cleaned_text = remove_stopwords(lemmatized_text)
    return cleaned_text

if __name__ == "__main__":    
    str_doc=lemmatize("<br>Asiakirja<br> <br><br> tekstistä pitää poistaa 10kg 10cm mitat.")
    print(str_doc)