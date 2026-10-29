import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy
import os

# On charge le tokenizer de spaCy (le petit modèle anglais)
try:
    nlp = spacy.load("en_core_web_sm") # permet de découper le texte en tokens intelligemment
except OSError:
    print("Attention : spaCy n'a pas trouvé 'en_core_web_sm'. Lance : python -m spacy download en_core_web_sm")
    raise

# 1- classe qui gere le vocabulaire (dictionnaire mot -> id et id -> mot)
class Vocab:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.sep_token = "<SEP>"
        
        # On initialise le dictionnaire avec nos tokens spéciaux
        self.word2id = {
            self.pad_token: 0, 
            self.unk_token: 1, 
            self.start_token: 2, 
            self.end_token: 3,
            self.sep_token: 4
        }
        self.id2word = {v: k for k, v in self.word2id.items()} # permettra de faire l'inverse (id -> mot)
        
    def build_vocab(self, texts, min_freq=2):
        """Construit le vocabulaire à partir d'une liste de textes."""
        counter = Counter()
        for text in texts:
            # spaCy découpe le texte intelligemment 
            tokens = [token.text.lower() for token in nlp.tokenizer(text)] 
            counter.update(tokens)
            
        idx = 5 # on commence à 5 car 0 à 4 sont pris par les tokens speciaux
        for word, freq in counter.most_common():
            if freq >= min_freq and word not in self.word2id:
                self.word2id[word] = idx
                self.id2word[idx] = word
                idx += 1
                
    def encode(self, text):
        """Transforme un texte en liste d'IDs."""
        tokens = [token.text.lower() for token in nlp.tokenizer(text)]
        return [self.word2id.get(token, self.word2id[self.unk_token]) for token in tokens]
        
    def decode(self, ids):
        """Transforme une liste d'IDs en texte lisible."""
        return " ".join([self.id2word.get(i, self.unk_token) for i in ids])


# --- 2. Classe Dataset PyTorch ---
class QGDataset(Dataset):
    def __init__(self, hf_dataset, vocab, max_src_len=100, max_tgt_len=20):
        self.data = hf_dataset
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # SQuAD peut avoir plusieurs réponses, on prend la première
        answer = item['answers']['text'][0]
        
        # Créer l'Input (Source) : Contexte + <SEP> + Réponse
        src_text = f"{item['context']} {self.vocab.sep_token} {answer}"
        
        # Créer la Target : <START> + Question + <END>
        tgt_text = f"{self.vocab.start_token} {item['question']} {self.vocab.end_token}"
        
        # Encoder (transformer en nombres)
        src_ids = self.vocab.encode(src_text)
        tgt_ids = self.vocab.encode(tgt_text)
        
        # Tronquer (Truncation) si c'est trop long
        src_ids = src_ids[:self.max_src_len]
        tgt_ids = tgt_ids[:self.max_tgt_len]
        
        # Retourner des tenseurs PyTorch
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


# --- 3. Fonction de Padding (Collate) ---
def collate_fn(batch):
    """
    Prend un batch de phrases de longueurs différentes et ajoute des <PAD> 
    pour que toutes les phrases d'un batch aient la même taille.
    """
    src_batch, tgt_batch = zip(*batch)
    
    # pad_sequence ajoute automatiquement la valeur 0 (<PAD>) à la fin
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0, batch_first=True)
    
    return src_padded, tgt_padded


# --- 4. Bloc de test (s'exécute seulement si on lance 'python src/dataset.py') ---
if __name__ == "__main__":
    from datasets import load_dataset
    
    print("Test du fichier dataset.py...")
    # On calcule le chemin absolu vers le dossier data pour éviter les erreurs
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    raw_train = load_dataset("rajpurkar/squad", split="train[:100]", cache_dir=data_dir)
    
    vocab = Vocab()
    all_texts = [row['context'] + " " + row['question'] + " " + row['answers']['text'][0] for row in raw_train]
    vocab.build_vocab(all_texts)
    
    dataset = QGDataset(raw_train, vocab)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    for src, tgt in loader:
        print("Test réussi ! Shapes :", src.shape, tgt.shape)
        print("Exemple source décodée :", vocab.decode(src[0].tolist())[:100], "...")
        break