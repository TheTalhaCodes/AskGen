import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import time
import os
import sys

# Permet d'importer les fichiers du dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import Vocab, QGDataset, collate_fn
from src.model import Encoder, Attention, Decoder, Seq2Seq

# ==========================================
# 1. HYPERPARAMÈTRES
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation de l'appareil : {DEVICE}")

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.2
TEACHER_FORCING_RATIO = 0.5

# Limite de mots par phrase (doit correspondre à ton dataset.py)
MAX_SRC_LEN = 100
MAX_TGT_LEN = 20

# Nom du fichier de sauvegarde
MODEL_SAVE_PATH = "saved_model/askgen_lstm_attention.pth"

# ==========================================
# 2. CHARGEMENT DES DONNÉES
# ==========================================
print("Chargement du dataset...")
# On charge un sous-ensemble pour que l'entraînement soit raisonnable
raw_train = load_dataset("rajpurkar/squad", split="train[:5000]", cache_dir="./data")
raw_val = load_dataset("rajpurkar/squad", split="validation[:1000]", cache_dir="./data")

print("Construction du vocabulaire...")
vocab = Vocab()
all_texts = [row['context'] + " " + row['question'] + " " + row['answers']['text'][0] for row in raw_train]
vocab.build_vocab(all_texts, min_freq=2)

print(f"Taille du vocabulaire : {len(vocab.word2id)} mots")

print("Création des DataLoaders...")
train_dataset = QGDataset(raw_train, vocab, max_src_len=MAX_SRC_LEN, max_tgt_len=MAX_TGT_LEN)
val_dataset = QGDataset(raw_val, vocab, max_src_len=MAX_SRC_LEN, max_tgt_len=MAX_TGT_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ==========================================
# 3. INITIALISATION DU MODÈLE
# ==========================================
print("Initialisation du modèle...")
vocab_size = len(vocab.word2id)

encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
attention = Attention(HIDDEN_DIM)
decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM, attention, NUM_LAYERS, DROPOUT)

model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

# ==========================================
# 4. FONCTION DE PERTE ET OPTIMISEUR
# ==========================================
# ignore_index=0 permet d'ignorer les <pad> dans le calcul de l'erreur
criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2id["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 5. BOUCLE D'ENTRAÎNEMENT
# ==========================================
def train_epoch(model, dataloader, optimizer, criterion, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    
    # tqdm affiche une barre de progression
    progress_bar = tqdm(dataloader, desc="Entraînement")
    
    for src, tgt in progress_bar:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt, teacher_forcing_ratio)
        
        # On enlève le premier token <start> pour comparer
        output = output[:, 1:].reshape(-1, output.shape[-1])
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        
        # Backward pass
        loss.backward()
        
        # Évite l'explosion des gradients (gradient clipping)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_loss += loss.item()
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Validation"):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            
            # Pas de teacher forcing en validation (on force à 0)
            output = model(src, tgt, teacher_forcing_ratio=0)
            
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)

# ==========================================
# 6. LANCEMENT
# ==========================================
if __name__ == "__main__":
    print("\nDébut de l'entraînement ! 🚀")
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, TEACHER_FORCING_RATIO)
        val_loss = evaluate(model, val_loader, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        print(f"\nÉpoque {epoch+1}/{NUM_EPOCHS} | Temps: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Sauvegarde du modèle
        os.makedirs("saved_model", exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_word2id': vocab.word2id,
            'vocab_id2word': vocab.id2word,
            'hyperparameters': {
                'embed_dim': EMBED_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS
            }
        }, MODEL_SAVE_PATH)
        print(f"Modèle sauvegardé dans {MODEL_SAVE_PATH} ✅\n")
        
    print("Entraînement terminé ! 🎉")