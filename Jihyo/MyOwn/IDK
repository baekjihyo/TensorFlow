from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

pipe = pipeline("text-classification", model="ProsusAI/finbert")

test = pipe("I was waiting for this moment for a long time. I am so happy to share this with you all.")

print(test)

model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

res = classifier("I was waiting for this moment for a long time. I am so happy to share this with you all.")

print(res)