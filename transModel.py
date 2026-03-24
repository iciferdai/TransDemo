from data_dict import *
from myTrans.enc_layer import *
from myTrans.dec_layer import *
from myTrans.pos import *

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_layer1 = DecoderLayer()
        self.decoder_layer2 = DecoderLayer()
        self.decoder_layer3 = DecoderLayer()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
        self.pos_embedding = PosEncoding()

    def forward(self, x, enc_o, src_mask=None, tgt_mask=None):
        # 1
        x = self.embedding(x)
        x = self.pos_embedding(x)
        # 2
        x, a_w1, c_w1 = self.decoder_layer1(x, enc_o, src_mask=src_mask, tgt_mask=tgt_mask)
        x, a_w2, c_w2 = self.decoder_layer2(x, enc_o, src_mask=src_mask, tgt_mask=tgt_mask)
        x, a_w3, c_w3 = self.decoder_layer3(x, enc_o, src_mask=src_mask, tgt_mask=tgt_mask)
        return x, a_w1, a_w2, c_w1, c_w2

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer1 = EncoderLayer()
        self.encoder_layer2 = EncoderLayer()
        self.encoder_layer3 = EncoderLayer()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
        self.pos_embedding = PosEncoding()

    def forward(self, x, mask=None):
        # 1.
        x= self.embedding(x)
        # 2.
        x = self.pos_embedding(x)
        # 3.
        x, w1 = self.encoder_layer1(x, mask=mask)
        x, w2 = self.encoder_layer2(x, mask=mask)
        x, w3 = self.encoder_layer3(x, mask=mask)
        return x, w1, w2

class MyTransf(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        self.fc = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        y, w1, w2 = self.enc(src, mask=src_mask)
        o, a_w1, c_w1, a_w2, c_w2 = self.dec(tgt, y, src_mask=src_mask, tgt_mask=tgt_mask)
        o = self.fc(o)
        return o, w1, w2, a_w1, c_w1, a_w2, c_w2