
from nltk.translate.meteor_score import meteor_score
from nltk.translate.gleu_score import sentence_gleu
from nltk.tokenize import word_tokenize
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from pydantic import BaseModel
from pydantic import computed_field
from time import sleep
import torch

class NLTKCalculator(BaseModel):
    reference: str
    prediction: str

    @computed_field
    @property
    def cleaned_sentences(self) -> dict:

        return {
            "reference": self.reference.lower().strip(),
            "prediction": self.prediction.lower().strip(),
        }

    def meteor(self) -> float:
        return meteor_score([word_tokenize(self.cleaned_sentences["reference"])], word_tokenize(self.cleaned_sentences["prediction"]))

    def gleu(self) -> float:
        return sentence_gleu([(self.reference)], self.prediction)

    def bert(self) -> float:
        sleep(0.5)  # Sleep to avoid rate limiting issues with BERTScorer

        _, _, F = BERTScorer(model_type="bert-base-uncased", lang="en").score([self.cleaned_sentences["prediction"]], [self.cleaned_sentences["reference"]])

        if isinstance(F, torch.Tensor):
            return round(F.mean().item(), 4)
        
        # Return the mean F1 score rounded to 4 decimal places
        # Note: F1 is a tensor, so we convert it to a float and round
        return 0.0
        # Note: F1 is a tensor, so we convert it to a float and round

    def rouge(self) -> float:
        # Initialize ROUGE scorer with ROUGE-L metric
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Calculate ROUGE scores
        scores = scorer.score(self.cleaned_sentences["reference"], self.cleaned_sentences["prediction"])
        
        # Return ROUGE-L F1 score (most commonly used)
        return round(scores['rougeL'].fmeasure, 4)


if __name__ == "__main__":
    # Example usage
    reference = "The cat sat on the mat"
    prediction = "The cat sat on the floor"

    nltk_calculator = NLTKCalculator(reference=reference, prediction=prediction)
    score = nltk_calculator.meteor()
    gleu_score = nltk_calculator.gleu()
    bert_score = nltk_calculator.bert()
    rouge_score = nltk_calculator.rouge()

    print(f"METEOR score: {score:.4f}")
    print(f"GLEU score: {gleu_score:.4f}")
    print(f"BERT score: {bert_score:.4f}")
    print(f"ROUGE score: {rouge_score:.4f}")
