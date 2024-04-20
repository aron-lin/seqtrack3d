''' Define the seq2seq model '''
import torch
import torch.nn as nn
from .Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F

from .pytorch_utils import Seq

class Encoder(nn.Module):
    
    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos


    def forward(self, src_seq, src_mask=None, return_attns=False, global_feature=False):
        
        enc_slf_attn_list = []
        # -- Forward
        if global_feature:
            enc_output = self.dropout(self.with_pos_embed(src_seq)) #--positional encoding off
        else:
            enc_output = self.dropout(self.with_pos_embed(src_seq)) 

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask) # vanilla attention mechanism
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list


        return enc_output,


class Decoder(nn.Module):

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = (trg_seq)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output, dec_enc_attn_list

class Seq2SeqFormer(nn.Module):
    """
    A sequence-to-sequence transformer model that facilitates deep interaction between 
    point cloud sequences and bounding box (bbox) sequences through an attention-based mechanism.
    This leverages the inherent spatial and temporal relationships within the sequences to 
    enhance feature representation for tasks involving point clouds and their associated bounding boxes.
    """

    def __init__(
            self, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=64, d_model=64, d_inner=512,
            n_layers=3, n_head=8, d_k=32, d_v=32, dropout=0.2, n_position=100):

        super().__init__()
        
        self.d_model=d_model
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.proj=nn.Linear(128,d_model) 
        self.proj2=nn.Linear(4,d_model) # 4 represents the dimensions for x, y, z, plus a time stamp
        self.l1=nn.Linear(d_model*8, d_model)
        self.l2=nn.Linear(d_model, 4)

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)
        
        self.encoder_global = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.decoder = Decoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, trg_seq,src_seq,valid_mask):

        src_seq_=self.proj(src_seq) # Adjust the input features to 128 dimensions
        trg_seq_=self.proj2(trg_seq) # Also adjust Q to 128 dimensions, corresponding to the features of the input box

        enc_output, *_ = self.encoder(src_seq_.reshape(-1,128,self.d_model)) # Locally apply self-attention to every single frame

        enc_others,*_=self.encoder_global(src_seq_, global_feature=True) # Apply attention across frames globally

        # Implementing cross-decoder
        # Q: trg_seq_
        # K, V: Concatenate(enc_output, enc_others)
        enc_output=torch.cat([enc_output.reshape(-1,4*128,self.d_model),enc_others],dim=1) # default 4 frames
        dec_output, dec_attention,*_ = self.decoder(trg_seq_, None, enc_output, None) 
                                                

        # Project to output
        dec_output=dec_output.view(dec_output.shape[0],4,self.d_model*8)
        dec_output= self.l1(dec_output)
        dec_output= self.l2(dec_output)
        
        return dec_output