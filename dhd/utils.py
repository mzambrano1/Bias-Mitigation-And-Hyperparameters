from sklearn.model_selection import KFold
from operator import itemgetter

def folds(n,n_splits=5):
    kf = KFold(n_splits=n_splits,random_state=42,shuffle=True)
    
    folds = enumerate(kf.split(list(range(0,n))))
    
    folds = list(folds)

        
    return folds


def words_from_index(indexes, words):
    return list(itemgetter(*indexes)(words))

def make_def_pairs(male,female):
    def_pairs = []
    for i in range(len(male)):
        def_pairs.append([female[i],male[i]])
    return def_pairs