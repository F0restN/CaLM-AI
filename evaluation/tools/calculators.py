from math import log2
from typing import Any

import evaluate
from pydantic import BaseModel, field_validator, model_validator

## ROUGE Evaluation

class EvalCalculatorFactory(BaseModel):
    predictions: Any
    references: Any

    @field_validator("predictions", "references", mode="after")
    @classmethod
    def validate_inputs(cls, value: list[str] | str) -> list[str]:
        """Validate the inputs."""
        if not isinstance(value, list):
            return [value]
        return value

    def rouge(self) -> dict | None:
        """Calculate the ROUGE score for a list of predictions and references.

        Returns:
            json: {
                "rouge1": float
                "rouge2": float
                "rougeL": float
                "rougeLsum": float
            }

        """
        rouge = evaluate.load("rouge")
        return rouge.compute(predictions=self.predictions, references=self.references)


    ## BLEU Evaluation
    def bleu(self) -> dict | None:
        """Calculate the BLEU score for a list of predictions and references.

        Returns:
            json: {
                "bleu": float
                "precisions": List[float]
                "brevity_penalty": float
            }

        """
        bleu = evaluate.load("bleu")
        return bleu.compute(predictions=self.predictions, references=self.references)


    ## CHAR F Score Evaluation
    def char_f(self) -> dict | None:
        """Calculate the CHAR F score for a list of predictions and references.

        Returns:
            json: {
                "score": float,
                "char_order": int,
                "word_order": int,
                "beta": int
            }

        """
        chrf = evaluate.load("chrf")
        return chrf.compute(predictions=self.predictions, references=self.references)


    ## Calculate BERT Score
    def bert(self, model_type: str = "microsoft/deberta-xlarge-mnli") -> dict | None:
        """Calculate the BERT score for a list of predictions and references.

        Returns:
            json: {
                "precision": [float, float],
                "recall": [float, float],
                "f1": [float, float],
                "hashcode": str
            }

        """
        bert = evaluate.load("bertscore")
        return bert.compute(predictions=self.predictions, references=self.references, model_type=model_type, device="cuda")


class RecallCalculatorFactory(BaseModel):
    predictions: Any
    references: Any

    @field_validator("predictions", "references", mode="after")
    @classmethod
    def validate_inputs(cls, value: list[str] | str) -> list[str]:
        """Validate the inputs."""
        if not isinstance(value, list):
            return [value]
        return value

    @model_validator(mode="after")
    def validate_non_empty(self) -> "RecallCalculatorFactory":
        """Validate that predictions and references are not empty."""
        if not self.predictions or not self.references:
            raise ValueError("Predictions and references cannot be empty")
        return self

    def recall(self, k: int | None = None) -> float | None:
        """Calculate recall at k for predictions and references."""
        preds = self.predictions[:k] if k is not None else self.predictions
        pred_set = set(preds)
        act_set = set(self.references)
        return len(act_set & pred_set) / len(act_set) if act_set else None

    def mrr(self) -> float | None:
        """Calculate the Mean Reciprocal Rank (MRR) for predictions and references."""
        reference_set = set(self.references)
        for i, pred in enumerate(self.predictions):
            if pred in reference_set:
                return 1.0 / (i + 1)
        return None

    def average_precision(self, k: int | None = None) -> float | None:
        """Calculate the Average Precision at K (AP@K) for predictions and references."""
        preds = self.predictions[:k] if k is not None else self.predictions
        reference_set = set(self.references)
        relevant_count = 0
        sum_precision = 0.0
        for i, pred in enumerate(preds):
            if pred in reference_set:
                relevant_count += 1
                sum_precision += relevant_count / (i + 1)
        return round(sum_precision / relevant_count, 2) if relevant_count > 0 else None

    def dcg(self, relevancy: list[float], k: int | None = None) -> float | None:
        """Calculate the Discounted Cumulative Gain (DCG) at K for predictions and relevancy scores."""
        if not relevancy:
            return None
        rel = relevancy[:k] if k is not None else relevancy
        dcg = sum(score / log2(i + 2) for i, score in enumerate(rel) if score != 0)
        return round(dcg, 2)

    def ndcg(self, relevancy: list[float], ideal_relevancy: list[float], k: int | None = None) -> float | None:
        """Calculate the Normalized Discounted Cumulative Gain (NDCG) at K for predictions and relevancy scores."""
        if not relevancy or not ideal_relevancy:
            return None
        rel = relevancy[:k] if k is not None else relevancy
        ideal_rel = ideal_relevancy[:k] if k is not None else ideal_relevancy
        dcg_val = self.dcg(rel)
        idcg_val = self.dcg(ideal_rel)
        return round(dcg_val / idcg_val, 2) if idcg_val and dcg_val is not None else None


if __name__ == "__main__":

    pred = ["I'm happy"]
    ref = ["He's happy"]

    calculator = EvalCalculatorFactory(predictions=pred, references=ref)

    print(calculator.bert())
