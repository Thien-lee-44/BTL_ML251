import re
import pickle
import torch
import numpy as np
from wordcloud import STOPWORDS
from transformers import AutoTokenizer, AutoModel,AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report


import matplotlib.pyplot as plt
import seaborn as sns

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)

def loader(path):
    X=np.load(path+"/data.npy")
    y=np.load(path+"/label.npy")
    with open(path+"/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return X,y,le
class PoemDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item   
def BERT(data,label,save=True): 
    le= LabelEncoder()
    label = torch.tensor(le.fit_transform(label))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model =AutoModel.from_pretrained(
        "bert-base-uncased",
        num_labels=5,
    )
    encodings = tokenizer(
        list(data),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    dataset = PoemDataset(encodings,  label)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_embeddings = []
    all_labels = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels_batch = batch["labels"]
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels_batch)
    X = torch.cat(all_embeddings, dim=0)
    y = torch.cat(all_labels, dim=0)
    if save:
        np.save("features/BERT/data.npy",X)
        np.save("features/BERT/label.npy",y)
        with open("features/BERT/label_encoder.pkl", "wb") as f: pickle.dump(le, f)
    return X,y,le

def report(all_preds,all_labels,label_encoding):
    cm = confusion_matrix(all_labels, all_preds)
    classes = label_encoding.classes_

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    print("\n Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=classes))

def simpleModel(data,label,label_encoding,test_size=0.2,random=42):
    models_list = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM":LinearSVC(),
        "Naive Bayes": GaussianNB()
    }
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size, random_state=random)
    for name, model in models_list.items():
        print("Training "+ name)
        model.fit(X_train, y_train)
        all_preds = model.predict(X_test)
        all_labels = y_test
        report(all_preds,all_labels,label_encoding)

    
def finetuneBERT(data,label):
    le= LabelEncoder()
    label = torch.tensor(le.fit_transform(label))
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model =AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=5,
    )
    train_encodings = tokenizer(
        list(X_train),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
   
    train_dataset = PoemDataset(train_encodings,  y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done, loss = {loss.item():.4f}")
    
    model.eval()
    test_encodings = tokenizer(
        list(X_test),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    all_preds = []
    all_labels = y_test
    with torch.no_grad():
        logits = model(**test_encodings).logits
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
    report(all_preds,all_labels,le)
  