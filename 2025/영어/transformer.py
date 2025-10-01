from transformers import pipeline
import torch

pipe = pipeline("text-generation", model="sshleifer/tiny-gpt2")

class PoliticalBiasAnalyzer:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        self.candidate_labels = ["conservative", "liberal"]
    
    def analyze_bias(self, sentence):
        result = self.classifier(
            sentence, 
            self.candidate_labels, 
            multi_label=False
        )
        
        bias_result = {}
        for label, score in zip(result['labels'], result['scores']):
            bias_result[label] = round(score * 100, 2)
        
        return bias_result

if __name__ == "__main__":
    analyzer = PoliticalBiasAnalyzer()
    
    test_sentences = [
        "The government should cut taxes to stimulate economic growth and empower small businesses.",
        "We need to increase public spending on healthcare and education to ensure social equality.",
        "The new policy aims to balance the budget by reducing unnecessary government programs."
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n문장 {i}: {sentence}")
        result = analyzer.analyze_bias(sentence)
        print(f"  정치적 편향성 분석 결과:")
        print(f"  보수(Conservative): {result['conservative']}%")
        print(f"  진보(Liberal): {result['liberal']}%")