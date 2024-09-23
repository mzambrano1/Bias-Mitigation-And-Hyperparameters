from gensim.models import KeyedVectors
from wefe.word_embedding_model import WordEmbeddingModel 
from sklearn.model_selection import KFold
from wefe.datasets import (
    load_weat,
    fetch_eds,
    fetch_debias_multiclass,
    fetch_debiaswe,
    load_bingliu,
)

from operator import itemgetter

from sklearn.model_selection import train_test_split
from wefe.debias.hard_debias import HardDebias
import pandas as pd

from metrics import *
from utils import *

print("Loading model...")
# Load the model
glove_wiki = KeyedVectors.load("../../../../ablation/original models/glove-wiki-gigaword-300.kv")#KeyedVectors.load("../glove-wiki-gigaword-300.kv")
model_glove = WordEmbeddingModel(glove_wiki,name="glove-wiki-gigaword-300")
print("Model loaded")
#define wordsets

fems = [
    'she', 'woman', 'female', 'femme', 'feminine', 'her', 'herself', 'lady', 'madam',
    'girl', 'gal', 'girlfriend', 'mother', 'mom', 'wife', 'grandmother', 
    'daughter', 'sister', 'aunt', 'niece', 'actress', 'mary', 'princess', 'queen', 
    'damsel', 'mademoiselle', 'countess', 'maiden', 'matron', 'matriarch', 
    'bride', 'ladyship', 'heiress', 'empress', 'governess', 'enchantress', 'goddess',
    "granddaughter","nun","duchess","hostess","sultana","fiancée","dame","priestess"

]

males = [
    'he', 'man', 'male', 'homme', 'masculine', 'him', 'himself', 'gentleman', 'sir', 
    'boy', 'guy', 'boyfriend', 'father', 'dad', 'husband', 'grandfather', 
    'son', 'brother', 'uncle', 'nephew', 'actor', 'john', 'prince', 'king', 
    'squire', 'monsieur', 'count', 'lad', 'patron', 'patriarch', 
    'groom', 'lordship', 'heir', 'emperor', 'governor', 'enchanter', 'god', "grandson","monk",
    "duke","host","sultan","fiance","gent","priest"]

debiaswe_wordsets = fetch_debiaswe()
gender_specific = debiaswe_wordsets["gender_specific"]

WEAT_wordsets = load_weat()

RND_wordsets = fetch_eds()
sentiments_wordsets = load_bingliu()
debias_multiclass_wordsets = fetch_debias_multiclass()

#Gender specific y attributes
esta= []
no_esta = []
for i in fems:
    if i in gender_specific:
        esta.append(i)
    else:
        no_esta.append(i)

for i in males:
    if i in gender_specific:
        esta.append(i)
    else:
        no_esta.append(i)
        
for i in no_esta:
    gender_specific.append(i)
    
    
A = [WEAT_wordsets["career"], WEAT_wordsets["family"],
WEAT_wordsets["math"], WEAT_wordsets["arts"],
WEAT_wordsets["science"], WEAT_wordsets["arts"],
RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_appearance"],
RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_sensitive"],
WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"],
sentiments_wordsets["positive_words"], sentiments_wordsets["negative_words"],
debias_multiclass_wordsets["male_roles"],debias_multiclass_wordsets["female_roles"]]

attributes = [item for sublist in A for item in sublist] 
attributes = list(set(attributes))
targets = attributes

esta = []
for i in gender_specific:
    if i in attributes:
        esta.append(i)

for i in esta:
    gender_specific.remove(i)
    


solver = ["auto", "full", "arpack", "randomized"]

outer_folds = folds(len(fems),10)

weats_bests = []
weates_bests = []
rnd_bests = []
rnsb_bests = []
ects_bests = []
ripa_bests = []

weats_all = []
weates_all = []
rnd_all = []
rnsb_all = []
ects_all = []
ripa_all = []

best_params = []

all_df = pd.DataFrame()
all_df2 = pd.DataFrame()

print("Starting CV...")

for index,fold in outer_folds:
    
    #se definen los folds outer 
    outer_train, outer_test = fold[0],fold[1]
    
    outer_train_f, outer_test_f = words_from_index(outer_train,fems), words_from_index(outer_test,fems)
    outer_train_m, outer_test_m = words_from_index(outer_train,males), words_from_index(outer_test,males)
    
    #se definen los folds inner para cada outer, para hacer la optimización
    inner_train_f, inner_test_f, inner_train_m, inner_test_m = train_test_split(outer_train_f,outer_train_m,test_size=0.5)
    
    #arreglos para guardar scores
    weats_scores = []
    ect_scores = []
    weates_scores = []
    ripa_scores = []
    rnsb_scores = []
    rnd_scores = []
    best_param =-1
    best_score = 1000
    
    for param in solver:
        
        #se buscan los mejores parametros para cada fold inner
        d_pairs = make_def_pairs(inner_train_m,inner_train_f)
        
        hd = HardDebias(pca_args = {'svd_solver': param}).fit(
        model=model_glove, definitional_pairs=d_pairs)
        
        debiased_model = hd.transform(model = model_glove,target=targets,ignore=gender_specific)
        
        ect, rnd, weates, ripa, rnsb, weat = run_metrics(debiased_model, outer_test_m,outer_test_f)
        weats_scores.append(weat)
        ect_scores.append(ect)
        weates_scores.append(weates)
        ripa_scores.append(ripa)
        rnsb_scores.append(rnsb)
        rnd_scores.append(rnd)
        df = pd.DataFrame(pd.DataFrame([['WEAT', weat,param], ['WEATES', weates,param],
                                       ['ECT', ect,param], ['RNSB', rnsb,param],
                                        ['RND', rnd,param], ['RIPA', ripa,param]],
                                        columns=['Metric','Value','solver']))
        df2 = pd.DataFrame({'Weat':weat,'Weates':weates,'RND':rnd,'RIPA':ripa,'ECT':ect,'RNSB':rnsb,'solver':param},index=[index])
        all_df = pd.concat([all_df,df])
        all_df2 = pd.concat([all_df2,df2])
        
        if ripa < best_score:
          best_score = ripa
          best_param = param
        
        all_df = pd.concat([all_df,df])
        
    weats_all.append(weats_scores) 
    weates_all.append(weates_scores)
    rnd_all.append(rnd_scores)
    rnsb_all.append(rnsb_scores)
    ects_all.append(ect_scores)
    ripa_all.append(ripa_scores)
    
    #se busca el mejor score
    best_params.append(best_param)
    
    # se entrena el modelo con los mejores parametros con el fold outer
    d_pairs = make_def_pairs(outer_train_m,outer_train_f)
    
    hd_best = HardDebias(pca_args = {'svd_solver': best_param}).fit(
    model=model_glove, definitional_pairs=d_pairs)
        
    debiased_model_best = hd_best.transform(model = model_glove,target=targets,ignore=gender_specific)
        
    weat = run_weat(debiased_model_best, outer_test_m,outer_test_f)
    weats_bests.append(weat)
    
    ects_bests.append(run_ect(debiased_model_best, outer_test_m,outer_test_f))
    weates_bests.append(run_weates(debiased_model_best, outer_test_m,outer_test_f))
    ripa_bests.append(run_ripa(debiased_model_best, outer_test_m,outer_test_f))
    rnsb_bests.append(run_rnsb(debiased_model_best, outer_test_m,outer_test_f))
    rnd_bests.append(run_rnd(debiased_model_best, outer_test_m,outer_test_f))
    
print("CV finished") 
print("Saving results...") 
bests_df = pd.DataFrame({'Weat':weats_bests,'Weates':weates_bests,'RND':rnd_bests,'RIPA':ripa_bests,'ECT':ects_bests,'RNSB':rnsb_bests,'solver':best_params})
    
bests_df.to_csv('./results/bests_hd_ripa.csv')
all_df.to_csv('./results/all_hd_ripa.csv')
all_df2.to_csv('./results/all_hd_ripa2.csv')
print("Results saved")