import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder (lit le contexte et la réponse )
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1,dropout=0.1):
        super().__init__()
        
        # couche d'embedding pour transformer les IDs en vecteurs denses
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM : lit la séquence mot par mot
        # batch_first=True signifie que l'input a la forme [batch_size, seq_len, embed_dim]
        self.lstm = nn.LSTM(embed_dim,hidden_dim, num_layers=num_layers, 
                            batch_first=True,bidirectional=True, dropout=dropout)
        
        # on multiplie par 2 car on utilise un LSTM bidirectionnel (Forward + Backward)
        self.fc=nn.Linear(hidden_dim*2,hidden_dim)
        
    
    def forward(self,src): # dimension de src : [batch_size, seq_len]
        
        embedded = self.embedding(src)  # retourne [batch_size, seq_len, embed_dim]
        
        outputs, (hidden, cell) = self.lstm(embedded) # le lstm renvoie tous etats cachés (outputs) et les etats finaux (hidden, cell)
        # dimension de outputs : [batch_size, seq_len, hidden_dim*2] (car bidirectionnel)
        
        # On combine les directions forward et backward pour l'état initial du décodeur
         # hidden shape: [num_layers * 2, batch_size, hidden_dim]
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]),dim=1)))
        cell = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]),dim=1)))
             # torch.tanh est unefonction d'activation qui permet de normaliser
            # les valeurs entre -1 et 1, ce qui aide à stabiliser l'apprentissage.
        
        return outputs, hidden.unsqueeze(0), cell.unsqueeze(0) # unsqueeze(0) ajoute une dimension pour à la position 0
    

class Attention(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        
        self.att = nn.Linear(hidden_dim*3,hidden_dim) # calcule 
        
    def forward(self,hidden,encoder_outputs): 
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1,src_len,1)
        
        energy = torch.tanh(self.att(torch.cat((hidden, encoder_outputs), dim=2)))    
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention,dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, attention, num_layers=1, dropout=0.1):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(embed_dim + hidden_dim*2, hidden_dim, num_layers=num_layers,
                             batch_first=True, dropout=dropout)
        
        self.fc_out = nn.Linear(hidden_dim*3, vocab_size) 
        
    def forward(self, input_token, hidden, cell, encoder_outputs):
        
        input_token = input_token.unsqueeze(1) # [batch_size] -> [batch_size, 1]
        embedded = self.embedding(input_token) # [batch_size, 1, embed_dim]
        
        a = self.attention(hidden.squeeze(0), encoder_outputs) # [batch_size, src_len]
        a = a.unsqueeze(1) # [batch_size, 1, src_len]
        
        weighted = torch.bmm(0, encoder_outputs)

        rnn_input = torch.car((embedded, weighted), dim=2) 
        
        output, (hidden,cell) = self.lstm(rnn_input, (hidden, cell))
        
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))
        
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        
        encoder_outputs, hidden, cell = self.encoder(src)
        
        input_token = tgt[:,0]
        
        for t in range(1, tgt_len):
            
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:,t,:] = output
            
            top1 = output.argmax(1)
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = tgt[:, t] if teacher_force else top1
            
        
        return outputs