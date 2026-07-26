"""
Phase B — Model
================
A small character-level seq2seq model with attention:
  Encoder: BiGRU over Bangla grapheme characters
  Decoder: GRU with Bahdanau attention over encoder states,
           conditioned additionally on a register embedding
           (native / tatsama / loanword / etc.) as an auxiliary
           input at every decoder step.

Why register conditioning matters (from your empirical findings):
  স -> s vs sh, and o vs O, are largely UNPREDICTABLE from spelling
  alone but correlate with etymological register. Feeding the register
  tag into the decoder gives the model a direct signal for exactly
  the cases your rule engine could not resolve.

This is intentionally small — a few hundred thousand parameters —
appropriate for a ~2,000-word dataset. Scale up once the lexicon grows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, emb_dim=64, hidden_dim=128, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True,
                           bidirectional=True)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, input_lens):
        # input_ids: (batch, src_len)
        embedded = self.embedding(input_ids)  # (batch, src_len, emb_dim)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # outputs: (batch, src_len, hidden_dim*2)

        # combine forward/backward final hidden states -> decoder init state
        hidden_fwd = hidden[0]  # (batch, hidden_dim)
        hidden_bwd = hidden[1]  # (batch, hidden_dim)
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (batch, hidden*2)
        return outputs, hidden_cat


class Attention(nn.Module):
    """Bahdanau-style additive attention."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        # decoder_hidden: (batch, hidden_dim)
        # encoder_outputs: (batch, src_len, hidden_dim*2)
        src_len = encoder_outputs.size(1)
        h = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch, src_len, hidden_dim)
        energy = torch.tanh(self.attn(torch.cat([h, encoder_outputs], dim=2)))
        scores = self.v(energy).squeeze(2)  # (batch, src_len)
        scores = scores.masked_fill(mask == 0, -1e10)
        return F.softmax(scores, dim=1)  # (batch, src_len)


class Decoder(nn.Module):
    def __init__(self, output_vocab_size, emb_dim=64, hidden_dim=128,
                 register_vocab_size=8, register_emb_dim=16, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, emb_dim, padding_idx=pad_id)
        self.register_embedding = nn.Embedding(register_vocab_size, register_emb_dim)
        self.attention = Attention(hidden_dim)

        # decoder GRU input: token embedding + register embedding + attended context
        gru_input_dim = emb_dim + register_emb_dim + hidden_dim * 2
        self.gru = nn.GRU(gru_input_dim, hidden_dim, batch_first=True)

        self.out = nn.Linear(hidden_dim + hidden_dim * 2 + emb_dim, output_vocab_size)
        self.hidden_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def init_hidden(self, encoder_hidden_cat):
        return torch.tanh(self.hidden_proj(encoder_hidden_cat)).unsqueeze(0)

    def forward_step(self, input_token, hidden, encoder_outputs, mask, register_ids):
        # input_token: (batch,) current decoder input token id
        # hidden: (1, batch, hidden_dim)
        embedded = self.embedding(input_token).unsqueeze(1)  # (batch,1,emb_dim)
        reg_emb = self.register_embedding(register_ids).unsqueeze(1)  # (batch,1,reg_emb_dim)

        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs, mask)  # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch,1,hidden*2)

        gru_input = torch.cat([embedded, reg_emb, context], dim=2)
        output, hidden = self.gru(gru_input, hidden)

        combined = torch.cat([output, context, embedded], dim=2).squeeze(1)
        logits = self.out(combined)  # (batch, output_vocab_size)
        return logits, hidden, attn_weights


class Seq2SeqG2P(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size,
                 emb_dim=64, hidden_dim=128,
                 register_vocab_size=8, pad_id_in=0, pad_id_out=0):
        super().__init__()
        self.encoder = Encoder(input_vocab_size, emb_dim, hidden_dim, pad_id_in)
        self.decoder = Decoder(output_vocab_size, emb_dim, hidden_dim,
                                register_vocab_size, pad_id=pad_id_out)
        self.pad_id_in = pad_id_in

    def make_mask(self, input_ids):
        return (input_ids != self.pad_id_in).long()

    def forward(self, input_ids, input_lens, output_ids, register_ids,
                teacher_forcing_ratio=0.5):
        batch_size = input_ids.size(0)
        max_out_len = output_ids.size(1)
        output_vocab_size = self.decoder.out.out_features

        encoder_outputs, hidden_cat = self.encoder(input_ids, input_lens)
        mask = self.make_mask(input_ids)
        hidden = self.decoder.init_hidden(hidden_cat)

        outputs = torch.zeros(batch_size, max_out_len, output_vocab_size,
                               device=input_ids.device)

        dec_input = output_ids[:, 0]  # SOS token for every item
        for t in range(1, max_out_len):
            logits, hidden, _ = self.decoder.forward_step(
                dec_input, hidden, encoder_outputs, mask, register_ids
            )
            outputs[:, t, :] = logits

            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            dec_input = output_ids[:, t] if use_teacher_forcing else top1

        return outputs

    @torch.no_grad()
    def greedy_decode(self, input_ids, input_lens, register_ids,
                       sos_id, eos_id, max_len=30):
        """Inference: no teacher forcing, greedy argmax decoding."""
        self.eval()
        batch_size = input_ids.size(0)
        encoder_outputs, hidden_cat = self.encoder(input_ids, input_lens)
        mask = self.make_mask(input_ids)
        hidden = self.decoder.init_hidden(hidden_cat)

        dec_input = torch.full((batch_size,), sos_id, dtype=torch.long,
                                device=input_ids.device)
        results = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_len):
            logits, hidden, _ = self.decoder.forward_step(
                dec_input, hidden, encoder_outputs, mask, register_ids
            )
            top1 = logits.argmax(1)
            for i in range(batch_size):
                if not finished[i]:
                    tok = top1[i].item()
                    if tok == eos_id:
                        finished[i] = True
                    else:
                        results[i].append(tok)
            dec_input = top1
            if all(finished):
                break

        return results
