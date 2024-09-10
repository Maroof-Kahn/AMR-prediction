# AMR-prediction
# Antimicrobial resistance prediction using Machine Learning
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DNABertForSequenceClassification, DNABertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

# Step 1: Data Preparation
# Read data from CSV file
csv_file_path = "D:\dna bert\train.csv"
genome_df = pd.read_csv(csv_file_path)

# Split data into train and test sets
train_df, test_df = train_test_split(genome_df, test_size=0.2, random_state=42)

# Initialize tokenizer and tokenize the sequences
tokenizer = DNABertTokenizer.from_pretrained('NucleBERT/dna_bert_large')
train_encodings = tokenizer(train_df['sequence'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_df['sequence'].tolist(), truncation=True, padding=True)

# Convert labels to tensors
train_labels = torch.tensor(train_df['label'].tolist())  # Assuming 'label' column contains labels
test_labels = torch.tensor(test_df['label'].tolist())

# Step 2: Model Training
# Load pre-trained DNABERT-2 model for sequence classification
model = DNABertForSequenceClassification.from_pretrained('NucleBERT/dna_bert_large', num_labels=num_labels)

# Define TrainingArguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=test_encodings,
)

# Train the model
trainer.train()

# Step 3: Model Evaluation
# Evaluate the model on the test set
predictions = trainer.predict(test_encodings)
predicted_class = predictions.predictions.argmax(axis=1)
accuracy = accuracy_score(test_labels, predicted_class)
print("Accuracy:", accuracy)
