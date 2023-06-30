import model
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def define_simple_decoder(hidden_size, input_vocab_len, output_vocab_len, max_length):

    decoder = DecoderRNN(hidden_size,output_vocab_len)

    return decoder

def run_simple_decoder(simple_decoder, decoder_input, encoder_hidden, decoder_hidden, encoder_outputs):
    results = simple_decoder.forward(decoder_input,decoder_hidden)

    return results  


class BidirectionalEncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(BidirectionalEncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, 1, True, True, 0, True)

    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1*2, 1, self.hidden_size, device=device)

def define_bi_encoder(input_vocab_len, hidden_size):
    encoder = None

    encoder = BidirectionalEncoderRNN(input_vocab_len,hidden_size)

    return encoder

def fix_bi_encoder_output_dim(encoder_output, hidden_size):
    output = None

    output = encoder_output[:,:,:126] + encoder_output[:,:,126:]

    return output

def fix_bi_encoder_hidden_dim(encoder_hidden):

    output = encoder_hidden[0:1]

    return output


class AttnDecoderRNNDot(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNNDot, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)
        dot = torch.matmul(hidden[0], encoder_outputs.T)
        weights = F.softmax(dot, dim=1)    
        applied = torch.bmm(weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        o = torch.cat((embedded[0], applied[0]), 1)
        o = self.attn_combine(o).unsqueeze(0)
        o = F.relu(o)
        o, hidden = self.gru(o, hidden)
        o = F.log_softmax(self.out(o[0]), dim=1)
        return o, hidden, weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNNBilinear(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNNBilinear, self).__init__()

        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        weights = torch.nn.Linear(126, 126)
        transf = torch.matmul(weights(hidden[0]), encoder_outputs.T)
        weights_2 = F.softmax((transf), dim=1)
        applied = torch.bmm(weights_2.unsqueeze(0), encoder_outputs.unsqueeze(0))
        o = torch.cat((embedded[0], applied[0]), 1)
        o = self.attn_combine(o).unsqueeze(0)
        o = F.relu(o)
        o, hidden = self.gru(o, hidden)
        o = F.log_softmax(self.out(o[0]), dim=1)
        return o, hidden, weights_2
        

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
