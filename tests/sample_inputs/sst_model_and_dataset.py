import transformers

import codeattack

model_path = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

model = codeattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

dataset = codeattack.datasets.HuggingFaceDataset(
    "glue", subset="sst2", split="train", shuffle=False
)
