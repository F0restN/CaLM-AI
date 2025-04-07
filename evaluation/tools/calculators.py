import numpy as np

from typing import List
from math import log2
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import ndcg_score

## ROUGE Evaluation
def calculate_rouge_scores(predictions: List[str], references: List[str]):
    """
    Calculate the ROUGE score for a list of predictions and references.
    
    Args:
        predictions: List[str]
        references: List[str]
        
    Returns:
        json: {
            "rouge1": float
            "rouge2": float
            "rougeL": float
            "rougeLsum": float
        }
    """
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


## BLEU Evaluation
def calculate_bleu_score(predictions: List[str], references: List[str]):
    """
    Calculate the BLEU score for a list of predictions and references.
    
    Args:
        predictions: List[str]
        references: List[str]
        
    Returns:
        json: {
            "bleu": float
            "precisions": List[float]
            "brevity_penalty": float
        }
    """
    
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=predictions, references=references)
    

## Cosine Similarity Evaluation
def calculate_cosine_similarity(predictions: str, references: str):
    """
    Calculate the cosine similarity between a list of predictions and references.
    
    Args:
        predictions: List[str]
        references: List[str]
        
    Returns:
        float: The cosine similarity between the predictions and references.
    """
    
    sent_trans = SentenceTransformer("BAAI/bge-base-en-v1.5", device="mps")

    sim_score = util.pytorch_cos_sim(sent_trans.encode(predictions), sent_trans.encode(references))[0][0].item()
    
    return sim_score


## CHAR F Score Evaluation
def calculate_char_f_score(predictions: List[str], references: List[str]):
    """
    Calculate the CHAR F score for a list of predictions and references.
    
    Args:
        predictions: List[str]
        references: List[str]
        
    Returns:
        json: {
            "score": float,
            "char_order": int,
            "word_order": int,
            "beta": int
        }
    """
    chrf = evaluate.load("chrf")
    return chrf.compute(predictions=predictions, references=references)


## Calculate BERT Score
def calculate_bert_score(predictions: List[str], references: List[str], model_type: str = "microsoft/deberta-xlarge-mnli"):
    """
    Calculate the BERT score for a list of predictions and references.
    
    Args:
        predictions: List[str]
        references: List[str]
        
    Returns:
        json: {
            "precision": [float, float],
            "recall": [float, float],
            "f1": [float, float],
            "hashcode": str
        }
    """
    bert = evaluate.load("bertscore")
    return bert.compute(predictions=predictions, references=references, model_type=model_type, device="mps")


def calculate_recall(predictions: List[str], references: List[str], k: int):
    pred_set = set(predictions[:k])
    act_set = set(references)
    return len(act_set & pred_set) / float(len(act_set))
    

def calculate_mrr(predictions: List[str], references: List[str]) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) for a list of predictions and references.
    
    MRR is a statistic measure for evaluating any process that produces a list of possible responses to a sample of queries, ordered by probability of correctness.
    
    Args:
        predictions: List[str] - The list of predicted items
        references: List[str] - The list of reference (ground truth) items
        
    Returns:
        float: The Mean Reciprocal Rank score
    """
    if not predictions or not references:
        return 0.0
    
    # Convert references to a set for faster lookup
    reference_set = set(references)
    
    # Find the rank of the first correct prediction
    for i, pred in enumerate(predictions):
        if pred in reference_set:
            # MRR uses 1-based ranking, so add 1 to the index
            return 1.0 / (i + 1)
    
    # If no correct prediction is found
    return 0.0


def calculate_average_precision(predictions: List[str], references: List[str], k: int = None) -> float:
    """
    Calculate the Average Precision at K (AP@K) for a list of predictions and references.
    
    Average Precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each 
    threshold, with the increase in recall from the previous threshold used as the weight.
    
    Args:
        predictions: List[str] - The list of predicted items in ranked order
        references: List[str] - The list of reference (ground truth) items
        k: int, optional - The number of predictions to consider. If None, all predictions are considered.
        
    Returns:
        float: The Average Precision score at K
    """
    if not predictions or not references:
        return 0.0
    
    # Apply k limit if specified
    if k is not None:
        predictions = predictions[:k]
    
    # Convert references to a set for faster lookup
    reference_set = set(references)
    
    # Initialize variables
    relevant_count = 0
    sum_precision = 0.0
    
    # Calculate precision at each position where a relevant item is found
    for i, pred in enumerate(predictions):
        if pred in reference_set:
            relevant_count += 1
            precision_at_i = round((relevant_count / (i + 1)), 2)
            sum_precision += precision_at_i
    
    # If no relevant items were found in the predictions
    if relevant_count == 0:
        return 0.0
    
    # AP is the sum of precisions divided by the total number of relevant items
    return round((sum_precision / relevant_count), 2)


def calculate_dcg(predictions: List[str], relevancy: List[str], k: int = None) -> float:
    """
    Calculate the Discounted Cumulative Gain (DCG) at K for a list of predictions and references.
    
    DCG measures the usefulness, or gain, of a predicted item based on its position in the result list.
    The gain is accumulated from the top of the result list to the bottom, with the gain of each result
    discounted at lower ranks.
    
    Args:
        predictions: List[str] - The list of predicted items in ranked order
        references: List[str] - The list of reference (ground truth) items
        k: int, optional - The number of predictions to consider. If None, all predictions are considered.
        
    Returns:
        float: The DCG score at K
    """
    if not predictions or not relevancy:
        return 0.0
    
    # Apply k limit if specified
    if k is not None:
        predictions = predictions[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, pred in enumerate(predictions):
        # Check if prediction is relevant (exists in references)
        if relevancy[i] != 0:
            # Calculate gain using log2(i+1) to handle the case where i=0
            # The +2 is because we want log2(rank+1) and rank is i+1
            dcg += relevancy[i] / log2(i + 1 + 1)
        # print(f"DCG@{i+1} = {dcg}")
    
    return round(dcg, 2)


def calculate_ndcg(predictions: List[str], relevancy: List, ideal_relevancy: List, k: int = None) -> float:
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) at K for a list of predictions and references.
    
    NDCG normalizes the DCG by the ideal DCG, which is the DCG of the perfect ranking.
    
    Args:
        predictions: List[str] - The list of predicted items in ranked order
        references: List[str] - The list of reference (ground truth) items
        k: int, optional - The number of predictions to consider. If None, all predictions are considered.
        
    Returns:
        float: The NDCG score at K
    """
    if not predictions or not relevancy or not ideal_relevancy:
        return 0.0
    
    # Calculate DCG
    dcg = calculate_dcg(predictions, relevancy, k)
    
    idcg = calculate_dcg(predictions[:k], ideal_relevancy, k)
    
    # If IDCG is 0, return 0 to avoid division by zero
    if idcg == 0:
        return 0.0
    
    # Calculate NDCG
    ndcg = dcg / idcg
    
    return round(ndcg, 2)


if __name__ == "__main__":
    
    pred = [1,2,3]
    relevancy = [0,1,1]
    ideal_relevancy = [2,2,2]
    
    print(calculate_ndcg(pred, relevancy, ideal_relevancy, 3))
        