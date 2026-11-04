import torch
import os
import sys

# Permet d'importer les fichiers du dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import Vocab
from src.model import Encoder, Attention, Decoder, Seq2Seq

def load_model_for_generation(model_path, device):
    """Charge le modèle et le vocabulaire depuis un checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruire le vocabulaire
    vocab = Vocab()
    vocab.word2id = checkpoint['vocab_word2id']
    vocab.id2word = checkpoint['vocab_id2word']
    
    # Reconstruire l'architecture
    hp = checkpoint['hyperparameters']
    vocab_size = len(vocab.word2id)
    
    encoder = Encoder(vocab_size, hp['embed_dim'], hp['hidden_dim'], hp['num_layers'], 0.2)
    attention = Attention(hp['hidden_dim'])
    decoder = Decoder(vocab_size, hp['embed_dim'], hp['hidden_dim'], attention, hp['num_layers'], 0.2)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab

def generate_question(model, vocab, context, answer, max_len=20, device='cpu'):
    """Génère une question à partir d'un contexte et d'une réponse."""
    model.eval()
    
    # Troncature intelligente
    context_ids = vocab.encode(context)
    answer_ids = vocab.encode(answer)
    sep_id = vocab.word2id[vocab.sep_token]
    max_context_len = 100 - len(answer_ids) - 1
    context_ids = context_ids[:max_context_len]
    src_ids = context_ids + [sep_id] + answer_ids
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        input_token = torch.tensor([vocab.word2id[vocab.start_token]]).to(device)
        generated_ids = []
        
        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            top1 = output.argmax(1)
            word_id = top1.item()
            
            if word_id == vocab.word2id[vocab.end_token]:
                break
                
            generated_ids.append(word_id)
            input_token = top1
            
    return vocab.decode(generated_ids)