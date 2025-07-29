import torch
import torch.nn as nn
import numpy as np
from model.rotation2xyz import Rotation2xyz
from model.modules.transformer_modules import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer,  DenseFiLM, featurewise_affine, modulate
from model.modules.rotary_embedding_torch import RotaryEmbedding
from diffusion import logger
from einops import rearrange
from icecream import ic


class PDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, translation, pose_rep, glob, glob_rot,
                 latent_dim=512, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='spd',
                 arch='trans_enc', emb_trans_dec=False, norm_first=False, use_rotary=False, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'audio')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        ###### TODO: EDIT condition dimension ######
        self.instrument_cond_dim = 21
        # self.audio_cond_dim = 18
        # self.audio_cond_dim = 32
        if self.cond_mode == 'audio':
            self.audio_type = kargs.get('audio_type', 'jukebox')
            if self.audio_type == 'jukebox':
                self.audio_cond_dim = 4800
            elif self.audio_type == 'wav2clip':
                self.audio_cond_dim = 512
            else:
                raise TypeError('audio_type must be in ["jukebox", "wav2clip"]!')
            self.stft_dim = 2049
            self.mel_dim = 512
        
        ###### TODO: EDIT condition dimension ######
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        
        self.norm_first = norm_first
        if self.norm_first:
            logger.log('NORM FIRST!')
        self.use_rotary = use_rotary
        if self.use_rotary:
            logger.log('USE ROTERY POSITIONAL ENCODING!')
        
        if self.arch == 'trans_enc':
            logger.log("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        
        elif self.arch == 'trans_enc_film':
            logger.log("TRANS_ENC_FILM init")
            if self.use_rotary:
                self.rotary = RotaryEmbedding(dim=self.latent_dim, batch_first=False)
            else:
                self.rotary = None
            # self.seqTransRotEncoder = nn.Sequential()
            self.seqTransEncoder = TransformerEncoder()
            for _ in range(self.num_layers):
                self.seqTransEncoder.append(
                    TransformerEncoderLayer(
                        d_model=self.latent_dim,
                        nhead=self.num_heads,
                        dim_feedforward=self.ff_size,
                        dropout=self.dropout,
                        rotary=self.rotary,
                        norm_first=self.norm_first
                    )
                )
            self.film1 = DenseFiLM(self.latent_dim)
            self.film2 = DenseFiLM(self.latent_dim)
        
        elif self.arch == 'trans_dec_zero':
            logger.log("TRANS_DEC_ZERO init")
            if self.use_rotary:
                self.rotary = RotaryEmbedding(dim=self.latent_dim, batch_first=False)
            else:
                self.rotary = None
            self.seqTransDecoder = TransformerDecoder()
            for _ in range(self.num_layers):
                self.seqTransDecoder.append(
                    TransformerDecoderLayer(
                        d_model=self.latent_dim,
                        nhead=self.num_heads,
                        dim_feedforward=self.ff_size,
                        dropout=self.dropout,
                        rotary=self.rotary,
                        norm_first=self.norm_first
                    )
                )
            self.film1 = DenseFiLM(self.latent_dim)
            self.film2 = DenseFiLM(self.latent_dim)

            self.cond_encoder = nn.Sequential()
            for _ in range(2):
                self.cond_encoder.append(
                    nn.TransformerEncoderLayer(
                        d_model=latent_dim,
                        nhead=num_heads,
                        dim_feedforward=ff_size,
                        dropout=dropout,
                        activation=activation,
                        batch_first=False,
                    )
                )
            
            self.non_attn_cond_projection = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            )

            self.to_time_hidden = nn.Linear(latent_dim, latent_dim)
            self.to_time_tokens = nn.Linear(latent_dim, latent_dim)

        
        elif self.arch == 'trans_dec':
            logger.log("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'gru':
            logger.log("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, trans_dec_zero, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            # if 'text' in self.cond_mode:
            #     self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            #     print('EMBED TEXT')
            #     print('Loading CLIP...')
            #     self.clip_version = clip_version
            #     self.clip_model = self.load_and_freeze_clip(clip_version)
            # if 'action' in self.cond_mode:
            #     self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
            #     print('EMBED ACTION')
            if 'instrument' in self.cond_mode:
                self.embed_instrument = EmbedInstrument(self.instrument_cond_dim, self.latent_dim)
                logger.log('EMBED INSTRUMENT')
            if 'audio' in self.cond_mode:
                # self.embed_instrument = EmbedInstrument(self.instrument_cond_dim, self.latent_dim)
                self.embed_audio = EmbedAudio(self.audio_cond_dim, self.latent_dim)
                self.embed_stft = EmbedSTFT(self.stft_dim, self.latent_dim)
                self.embed_mel = EmbedMel(self.mel_dim, self.latent_dim)
                logger.log('EMBED AUDIO')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats, self.norm_first)

        self.rot2xyz = Rotation2xyz()

        self.norm = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.norm3 = nn.LayerNorm(latent_dim)
        self.norm_cond = nn.LayerNorm(latent_dim)

        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.embed_timestep.time_embed[0].weight, std=0.02)
        nn.init.normal_(self.embed_timestep.time_embed[2].weight, std=0.02)

        nn.init.normal_(self.embed_audio.audio_embedding.weight, std=0.02)

        nn.init.normal_(self.input_process.poseEmbedding.weight, std=0.02)

        nn.init.normal_(self.non_attn_cond_projection[1].weight, std=0.02)
        nn.init.normal_(self.non_attn_cond_projection[3].weight, std=0.02)

        nn.init.normal_(self.to_time_hidden.weight, std=0.02)
        nn.init.normal_(self.to_time_tokens.weight, std=0.02)

        for layer in self.seqTransDecoder.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, val=0)
            nn.init.constant_(layer.adaLN_modulation[-1].bias, val=0)
        
        nn.init.constant_(self.output_process.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.output_process.adaLN_modulation[-1].bias, 0)


    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond, force_mask=False, mask=None):
        nframes, bs, d = cond.shape  # embedding is permuted
        if force_mask:
            return torch.zeros_like(cond), None
        elif mask is not None:
            return cond * (1. - mask), None
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask), mask
        else:
            return cond, None
    
    def mask_hidden(self, hidden, mask, force_mask=False):
        bs, d = hidden.shape  # embedding is permuted
        if force_mask:
            return torch.zeros_like(hidden)
        elif self.training and self.cond_mask_prob > 0.:
            mask = mask.view(bs, 1)
            return hidden * (1. - mask)
        else:
            return hidden
            

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        t_emb = self.embed_timestep(timesteps)  # [1, bs, d]
        cond_emb = torch.zeros(nframes, bs, self.latent_dim)

        force_mask = y.get('uncond', False)
        
        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                enc_text = self.encode_text(y['text'])
            t_emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            t_emb += self.mask_cond(action_emb, force_mask=force_mask)
            
        if 'instrument' in self.cond_mode:
            instrument_emb = self.embed_instrument(y['instrument'])
            # emb += self.mask_cond(instrument_emb, force_mask=force_mask)
            instrument_emb = self.mask_cond(instrument_emb, force_mask=force_mask)
            cond_emb = instrument_emb

        if 'audio' in self.cond_mode:
            audio_emb = self.embed_audio(y['audio'])
            audio_emb = self.norm(audio_emb)
            cond_emb = audio_emb

            if 'mel' in y:
                mel_emb = self.embed_mel(y['mel'])
                mel_emb = self.norm2(mel_emb)

            if 'stft' in y:
                stft_emb = self.embed_stft(y['stft'])
                stft_emb = self.norm3(stft_emb)
                if self.arch == 'trans_enc_film':
                    cond_emb = torch.add(cond_emb, featurewise_affine(cond_emb, self.film2(stft_emb)))
                else:
                    cond_emb += stft_emb

        x = self.input_process(x)
        cond_emb = self.sequence_pos_encoder(cond_emb)
        cond_emb = self.cond_encoder(cond_emb)  # Trans Enc (nframes, bs, d)
        cond_emb, mask_emb = self.mask_cond(cond=cond_emb, force_mask=force_mask)

        mean_pooled_cond_emb = cond_emb.mean(dim=0)  # (bs, d)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_emb)  # [bs, d]
        cond_hidden = self.mask_hidden(cond_hidden, mask_emb, force_mask=force_mask)

        t_tokens = self.to_time_tokens(t_emb)
        t_cond = t_tokens.repeat(nframes, 1, 1)
        t_cond = t_cond + cond_emb

        t_emb = t_emb.squeeze()
        t_hidden = self.to_time_hidden(t_emb)
        t = t_hidden + cond_hidden

        if self.use_rotary:
            t_cond = self.rotary.rotate_queries_or_keys(t_cond)
        else:
            t_cond = self.sequence_pos_encoder(t_cond)

        # x = x + cond_emb

        if self.arch == 'trans_enc':
            # adding the timestep embed
            x = x + cond_emb
            xseq = torch.cat((t_emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
        
        elif self.arch == 'trans_enc_film':
            # adding the timestep embed
            # xseq = torch.cat((t_emb, x), axis=0)  # [seqlen+1, bs, d]
            # condseq = torch.cat((t_emb, cond_emb), axis=0)  # [seqlen+1, bs, d]
            if not self.use_rotary:
                x = self.sequence_pos_encoder(x)
            output = self.seqTransEncoder(x, cond=t_cond)  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            # xseq = x + t_emb  # [seqlen, bs, d]
            # output = self.seqTransEncoder(src=xseq, cond=cond_emb)

        elif self.arch == 'trans_dec_zero':
            if not self.use_rotary:
                x = self.sequence_pos_encoder(x)
            output = self.seqTransDecoder(x, memory=t_cond, t=t)

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((t_emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=t_emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=t_emb)
        
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        output = self.output_process(output, t)  # [bs, njoints, nfeats, nframes]
        return output


    # def _apply(self, fn):
    #     super()._apply(fn)
    #     self.rot2xyz.smpl_model._apply(fn)


    # def train(self, *args, **kwargs):
    #     super().train(*args, **kwargs)
    #     self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep == 'rot6d':
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats, norm_first):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.norm_first = norm_first
        self.norm = nn.LayerNorm(self.latent_dim)
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, 2 * latent_dim, bias=True)
        )

    def forward(self, output, cond):
        nframes, bs, d = output.shape
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        if self.norm_first:
            output = self.norm(output)
        output = modulate(output, shift, scale)
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        
        return output


class EmbedInstrument(nn.Module):
    def __init__(self, cond_dim, latent_dim):
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.instrument_embedding = nn.Linear(self.cond_dim, self.latent_dim)

    def forward(self, input):
        bs, nframes, npoints, nfeats = input.shape  # instrument: [bs, nframes, npoints, 3]
        input = input.reshape(bs, nframes, npoints*nfeats).float()  # [bs, nframes, cond_dim]
        assert(input.shape[-1] == self.cond_dim), f'input: {input.shape}, cond_dim: {self.cond_dim}'
        # print(input.dtype)  # torch.float32
        output = self.instrument_embedding(input)  # [bs, nframes, latent_dim]
        output = output.permute(1, 0, 2)  # [nframes, bs, latent_dim]
        
        return output


class EmbedAudio(nn.Module):
    def __init__(self, cond_dim, latent_dim):
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.audio_embedding = nn.Linear(self.cond_dim, self.latent_dim)

    def forward(self, input):
        bs, nframes, nfeats = input.shape
        input = input.float()
        assert(nfeats == self.cond_dim)
        # print(input.dtype)  # torch.float32
        output = self.audio_embedding(input)  # [bs, nframes, latent_dim]
        output = output.permute(1, 0, 2)  # [nframes, bs, latent_dim]
        
        return output
    
class EmbedSTFT(nn.Module):
    def __init__(self, cond_dim, latent_dim):
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.stft_embedding = nn.Linear(self.cond_dim, self.latent_dim)

    def forward(self, input):
        bs, nframes, nfeats = input.shape
        input = input.float()
        assert(nfeats == self.cond_dim)
        # print(input.dtype)  # torch.float32
        output = self.stft_embedding(input)  # [bs, nframes, latent_dim]
        output = output.permute(1, 0, 2)  # [nframes, bs, latent_dim]
        
        return output

class EmbedMel(nn.Module):
    def __init__(self, cond_dim, latent_dim):
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.mel_embedding = nn.Linear(self.cond_dim, self.latent_dim)

    def forward(self, input):
        bs, nframes, nfeats = input.shape
        input = input.float()
        assert(nfeats == self.cond_dim)
        # print(input.dtype)  # torch.float32
        output = self.mel_embedding(input)  # [bs, nframes, latent_dim]
        output = output.permute(1, 0, 2)  # [nframes, bs, latent_dim]
        
        return output
