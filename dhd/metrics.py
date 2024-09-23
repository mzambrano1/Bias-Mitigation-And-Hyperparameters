from wefe.datasets import fetch_debiaswe
from wefe.datasets import (
    load_weat,
    fetch_eds,
    fetch_debias_multiclass,
    fetch_debiaswe,
    load_bingliu,
)
from wefe.metrics import RNSB
from wefe.metrics import ECT
from wefe.metrics import RIPA
from wefe.metrics import RND
from wefe.query import Query
from wefe.utils import load_test_model
from wefe.metrics import WEAT
from wefe.utils import (
    run_queries,
)

debiaswe_wordsets = fetch_debiaswe()
definitional_pairs = debiaswe_wordsets["definitional_pairs"]
equalize_pairs = debiaswe_wordsets["equalize_pairs"]
gender_specific = debiaswe_wordsets["gender_specific"]

WEAT_wordsets = load_weat()

RND_wordsets = fetch_eds()
sentiments_wordsets = load_bingliu()
debias_multiclass_wordsets = fetch_debias_multiclass()


def run_weat(model, male, female):
    gender_1 = Query(


    [male, female],
    [WEAT_wordsets["career"], WEAT_wordsets["family"]],
    ["Male terms", "Female terms"],
    ["Career", "Family"],
    )

    gender_2 = Query(
    [male, female],
    [WEAT_wordsets["math"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Math", "Arts"],
    )

    gender_3 = Query(
    [male, female],
    [WEAT_wordsets["science"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Science", "Arts"],
    )

    gender_4 = Query(
    [male, female],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_appearance"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Appearence"],
    )

    gender_5 = Query(
    [male, female],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_sensitive"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Sensitive"],
    )

    gender_6 = Query(
    [male, female],
    [WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"]],
    ["Male terms", "Female terms"],
    ["Pleasant", "Unpleasant"],
    )

    gender_sent_1 = Query(
    [male, female],
    [sentiments_wordsets["positive_words"], sentiments_wordsets["negative_words"]],
    ["Male terms", "Female terms"],
    ["Positive words", "Negative words"],
    )

    gender_role_1 = Query(
    [male, female],
    [
    debias_multiclass_wordsets["male_roles"],
    debias_multiclass_wordsets["female_roles"],
    ],
    ["Male terms", "Female terms"],
    ["Man Roles", "Woman Roles"],
    )

    gender_queries = [
    gender_1,
    gender_2,
    gender_3,
    gender_4,
    gender_5,
    gender_6,
    gender_sent_1,
    gender_role_1,
    ]

    queries_sets = {

    'Gender' : gender_queries,
    }
    # load the model (in this case, the test model included in wefe)
    # instance the metric and run the query

    weat_scores = run_queries(
            WEAT,
            queries_sets['Gender'],
            [model],
            queries_set_name='Gender',
            aggregate_results=True,
            metric_params={"preprocessors": [{}, {"lowercase": True,}],},
            aggregation_function="abs_avg",
            warn_not_found_words=False,
        )
    return   weat_scores.iloc[-1, -1]
    

def run_rnsb(model, male, female):
    
    gender_1 = Query(


    [male, female],
    [WEAT_wordsets["career"], WEAT_wordsets["family"]],
    ["Male terms", "Female terms"],
    ["Career", "Family"],
    )

    gender_2 = Query(
    [male, female],
    [WEAT_wordsets["math"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Math", "Arts"],
    )

    gender_3 = Query(
    [male, female],
    [WEAT_wordsets["science"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Science", "Arts"],
    )

    gender_4 = Query(
    [male, female],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_appearance"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Appearence"],
    )

    gender_5 = Query(
    [male, female],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_sensitive"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Sensitive"],
    )

    gender_6 = Query(
    [male, female],
    [WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"]],
    ["Male terms", "Female terms"],
    ["Pleasant", "Unpleasant"],
    )

    gender_sent_1 = Query(
    [male, female],
    [sentiments_wordsets["positive_words"], sentiments_wordsets["negative_words"]],
    ["Male terms", "Female terms"],
    ["Positive words", "Negative words"],
    )

    gender_role_1 = Query(
    [male, female],
    [
    debias_multiclass_wordsets["male_roles"],
    debias_multiclass_wordsets["female_roles"],
    ],
    ["Male terms", "Female terms"],
    ["Man Roles", "Woman Roles"],
    )

    gender_queries = [
    gender_1,
    gender_2,
    gender_3,
    gender_4,
    gender_5,
    gender_sent_1,
    gender_role_1,
    ]

    queries_sets = {

    'Gender' : gender_queries,
    }
    # load the model (in this case, the test model included in wefe)
    # instance the metric and run the query
    RNSB_NUM_ITERATIONS = 100
    rnsb_scores = run_queries(
                RNSB,
                queries_sets['Gender'],
                [model],
                queries_set_name='Gender',
                metric_params={
                    "num_iterations": RNSB_NUM_ITERATIONS,
                    "preprocessors": [{}, {"lowercase": True,}],
                },
                aggregate_results=True,
                aggregation_function="abs_avg",
                warn_not_found_words=False,
            )
    return   rnsb_scores.iloc[-1, -1]
    

def run_ripa(model, male, female):
    
    gender_1 = Query(


    [male, female],
    [WEAT_wordsets["career"], WEAT_wordsets["family"]],
    ["Male terms", "Female terms"],
    ["Career", "Family"],
    )

    gender_2 = Query(
    [male, female],
    [WEAT_wordsets["math"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Math", "Arts"],
    )

    gender_3 = Query(
    [male, female],
    [WEAT_wordsets["science"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Science", "Arts"],
    )

    gender_4 = Query(
    [male, female],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_appearance"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Appearence"],
    )

    gender_5 = Query(
    [male, female],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_sensitive"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Sensitive"],
    )

    gender_6 = Query(
    [male, female],
    [WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"]],
    ["Male terms", "Female terms"],
    ["Pleasant", "Unpleasant"],
    )

    gender_sent_1 = Query(
    [male, female],
    [sentiments_wordsets["positive_words"], sentiments_wordsets["negative_words"]],
    ["Male terms", "Female terms"],
    ["Positive words", "Negative words"],
    )

    gender_role_1 = Query(
    [male, female],
    [
    debias_multiclass_wordsets["male_roles"],
    debias_multiclass_wordsets["female_roles"],
    ],
    ["Male terms", "Female terms"],
    ["Man Roles", "Woman Roles"],
    )

    gender_queries = [
    gender_1,
    gender_2,
    gender_3,
    gender_4,
    gender_5,
    gender_sent_1,
    gender_role_1,
    ]

    queries_sets = {

    'Gender' : gender_queries,
    }
    # load the model (in this case, the test model included in wefe)
    # instance the metric and run the query

    ripa_scores = run_queries(
                RIPA,
                queries_sets['Gender'],
                [model],
                queries_set_name='Gender',
                metric_params={
                    "preprocessors": [{}, {"lowercase": True,}],
                },
                generate_subqueries=True,
                aggregate_results=True,
                aggregation_function="abs_avg",
                warn_not_found_words=False,
            )
    return   ripa_scores.iloc[-1, -1]

def run_ect(model, male, female):
    
    gender_1 = Query(


    [male, female],
    [WEAT_wordsets["career"], WEAT_wordsets["family"]],
    ["Male terms", "Female terms"],
    ["Career", "Family"],
    )

    gender_2 = Query(
    [male, female],
    [WEAT_wordsets["math"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Math", "Arts"],
    )

    gender_3 = Query(
    [male, female],
    [WEAT_wordsets["science"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Science", "Arts"],
    )

    gender_4 = Query(
    [male, female],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_appearance"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Appearence"],
    )

    gender_5 = Query(
    [male, female],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_sensitive"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Sensitive"],
    )

    gender_6 = Query(
    [male, female],
    [WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"]],
    ["Male terms", "Female terms"],
    ["Pleasant", "Unpleasant"],
    )

    gender_sent_1 = Query(
    [male, female],
    [sentiments_wordsets["positive_words"], sentiments_wordsets["negative_words"]],
    ["Male terms", "Female terms"],
    ["Positive words", "Negative words"],
    )

    gender_role_1 = Query(
    [male, female],
    [
    debias_multiclass_wordsets["male_roles"],
    debias_multiclass_wordsets["female_roles"],
    ],
    ["Male terms", "Female terms"],
    ["Man Roles", "Woman Roles"],
    )

    gender_queries = [
    gender_1,
    gender_2,
    gender_3,
    gender_4,
    gender_5,
    gender_sent_1,
    gender_role_1,
    ]

    queries_sets = {

    'Gender' : gender_queries,
    }
    # load the model (in this case, the test model included in wefe)
    # instance the metric and run the query

 
    ect_scores = run_queries(
    ECT,
    queries_sets["Gender"],
    [model],
    queries_set_name='Gender',
    metric_params={
    "preprocessors": [{}, {"lowercase": True,}],
        },
    generate_subqueries=True,
    aggregate_results=True,
    aggregation_function="abs_avg",
    warn_not_found_words=False,
    )
    return  ect_scores.iloc[-1, -1]

def run_weates(model, male, female):
    
    gender_1 = Query(


    [male, female],
    [WEAT_wordsets["career"], WEAT_wordsets["family"]],
    ["Male terms", "Female terms"],
    ["Career", "Family"],
    )

    gender_2 = Query(
    [male, female],
    [WEAT_wordsets["math"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Math", "Arts"],
    )

    gender_3 = Query(
    [male, female],
    [WEAT_wordsets["science"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Science", "Arts"],
    )

    gender_4 = Query(
    [male, female],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_appearance"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Appearence"],
    )

    gender_5 = Query(
    [male, female],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_sensitive"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Sensitive"],
    )

    gender_6 = Query(
    [male, female],
    [WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"]],
    ["Male terms", "Female terms"],
    ["Pleasant", "Unpleasant"],
    )

    gender_sent_1 = Query(
    [male, female],
    [sentiments_wordsets["positive_words"], sentiments_wordsets["negative_words"]],
    ["Male terms", "Female terms"],
    ["Positive words", "Negative words"],
    )

    gender_role_1 = Query(
    [male, female],
    [
    debias_multiclass_wordsets["male_roles"],
    debias_multiclass_wordsets["female_roles"],
    ],
    ["Male terms", "Female terms"],
    ["Man Roles", "Woman Roles"],
    )

    gender_queries = [
    gender_1,
    gender_2,
    gender_3,
    gender_4,
    gender_5,
    gender_sent_1,
    gender_role_1,
    ]

    queries_sets = {

    'Gender' : gender_queries,
    }
    # load the model (in this case, the test model included in wefe)
    # instance the metric and run the query

 
    weat_es_scores = run_queries(
    WEAT,
    queries_sets["Gender"],
    [model],
    queries_set_name='Gender',
    metric_params={
        "return_effect_size": True,
        "preprocessors": [{}, {"lowercase": True,}],
        },
    aggregate_results=True,
    aggregation_function="abs_avg",
    warn_not_found_words=False,
    )
    return  weat_es_scores.iloc[-1, -1]

def run_rnd(model, male, female):
    
    gender_1 = Query(


    [male, female],
    [WEAT_wordsets["career"], WEAT_wordsets["family"]],
    ["Male terms", "Female terms"],
    ["Career", "Family"],
    )

    gender_2 = Query(
    [male, female],
    [WEAT_wordsets["math"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Math", "Arts"],
    )

    gender_3 = Query(
    [male, female],
    [WEAT_wordsets["science"], WEAT_wordsets["arts"]],
    ["Male terms", "Female terms"],
    ["Science", "Arts"],
    )

    gender_4 = Query(
    [male, female],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_appearance"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Appearence"],
    )

    gender_5 = Query(
    [male, female],
    [RND_wordsets["adjectives_intelligence"], RND_wordsets["adjectives_sensitive"]],
    ["Male terms", "Female terms"],
    ["Intelligence", "Sensitive"],
    )

    gender_6 = Query(
    [male, female],
    [WEAT_wordsets["pleasant_5"], WEAT_wordsets["unpleasant_5"]],
    ["Male terms", "Female terms"],
    ["Pleasant", "Unpleasant"],
    )

    gender_sent_1 = Query(
    [male, female],
    [sentiments_wordsets["positive_words"], sentiments_wordsets["negative_words"]],
    ["Male terms", "Female terms"],
    ["Positive words", "Negative words"],
    )

    gender_role_1 = Query(
    [male, female],
    [
    debias_multiclass_wordsets["male_roles"],
    debias_multiclass_wordsets["female_roles"],
    ],
    ["Male terms", "Female terms"],
    ["Man Roles", "Woman Roles"],
    )

    gender_queries = [
    gender_1,
    gender_2,
    gender_3,
    gender_4,
    gender_5,
    gender_sent_1,
    gender_role_1,
    ]

    queries_sets = {

    'Gender' : gender_queries,
    }
    # load the model (in this case, the test model included in wefe)
    # instance the metric and run the query

 
    rnd_scores = run_queries(
    RND,
    queries_sets['Gender'],
    [model],
    metric_params={"preprocessors": [{}, {"lowercase": True,}],},
    queries_set_name='Gender',
    aggregate_results=True,
    aggregation_function="abs_avg",
    generate_subqueries=True,
    warn_not_found_words=False,
    )        
    return  rnd_scores.iloc[-1, -1]


def run_metrics(model, male, female):
    metrics = [run_ect, run_rnd, run_weates, run_ripa, run_rnsb, run_weat]
    results = []
    for metric in metrics:
        results.append(metric(model,male,female))
    return results