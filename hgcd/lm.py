"""Thin HF-transformers wrapper: chat prompts, batched generation, KV-cached
candidate scoring, and contrastive (two-context) greedy decoding."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class LM:
    def __init__(self, name: str, device: Optional[str] = None, dtype=None,
                 device_map: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        cuda = torch.cuda.is_available()
        dtype = dtype or (torch.float16 if cuda else torch.float32)
        import transformers
        kw = dict(device_map=device_map) if device_map else {}
        major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        kw["dtype" if major_minor >= (4, 56) else "torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(name, **kw)
        if next(self.model.parameters()).dtype != dtype:  # never silently fall back to fp32
            self.model.to(dtype)
        if not device_map:
            self.model.to(device or ("cuda" if cuda else "cpu"))
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.n_boundary_fallback = 0
        cfg = self.model.config
        heads = getattr(cfg, "num_attention_heads", 16)
        kvh = getattr(cfg, "num_key_value_heads", None) or heads
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
        bytes_el = torch.finfo(dtype).bits // 8
        # KV-cache bytes per token: models without GQA (e.g. SmolLM2) need ~10x more than Qwen2.5
        self.kv_bytes_per_token = 2 * cfg.num_hidden_layers * kvh * hd * bytes_el
        self.vocab = getattr(cfg, "vocab_size", 150000)
        import os
        self.mem_budget = float(os.environ.get("HGCD_MEM_BUDGET_GB", "1.5")) * 2 ** 30

    def _safe_chunk(self, chunk: int, prefix_len: int, cand_len: int) -> int:
        kv = self.kv_bytes_per_token * (prefix_len + cand_len)
        logits = 4 * cand_len * self.vocab
        return max(1, min(chunk, int(self.mem_budget // max(1, kv + logits))))

    # ------------------------------------------------------------------ prompts
    def prompt(self, messages: List[dict], assistant_prefix: str = "", thinking: bool = False) -> str:
        kw = {}
        if "enable_thinking" in (getattr(self.tok, "chat_template", None) or ""):   # Qwen3 family
            kw["enable_thinking"] = bool(thinking)
        p = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        return p + assistant_prefix

    def _enc(self, texts: Sequence[str]):
        return self.tok(list(texts), return_tensors="pt", padding=True,
                        add_special_tokens=False).to(self.device)

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, prompts: List[str], max_new_tokens: int = 160, do_sample: bool = False,
                 temperature: float = 0.7, n: int = 1, batch_size: int = 8,
                 stop: str = "Answer:") -> List[str]:
        try:
            return self._generate(prompts, max_new_tokens, do_sample, temperature, n, batch_size, stop)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            torch.cuda.empty_cache()
            return self.generate(prompts, max_new_tokens, do_sample, temperature, n,
                                 max(1, batch_size // 2), stop)

    def _generate(self, prompts, max_new_tokens, do_sample, temperature, n, batch_size, stop):
        outs = []
        for i in range(0, len(prompts), batch_size):
            enc = self._enc(prompts[i:i + batch_size])
            kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      num_return_sequences=n, pad_token_id=self.tok.pad_token_id)
            if do_sample:
                kw.update(temperature=temperature, top_p=1.0, top_k=0)
            else:
                kw.update(temperature=None, top_p=None, top_k=None)
            try:
                g = self.model.generate(**enc, stop_strings=[stop], tokenizer=self.tok, **kw)
            except (TypeError, ValueError):
                g = self.model.generate(**enc, **kw)
            new = g[:, enc.input_ids.shape[1]:]
            for t in self.tok.batch_decode(new, skip_special_tokens=True):
                outs.append(t.split(stop)[0])
        return outs

    # ------------------------------------------------------------------ scoring
    def _cand_ids(self, p_ids: List[int], prefix: str, cands: Sequence[str]) -> List[List[int]]:
        out = []
        for c in cands:
            full = self.tok(prefix + c, add_special_tokens=False).input_ids
            if full[:len(p_ids)] == p_ids and len(full) > len(p_ids):
                out.append(full[len(p_ids):])
            else:  # BPE merge across the boundary: score the candidate tokenized alone
                self.n_boundary_fallback += 1
                out.append(self.tok(c, add_special_tokens=False).input_ids or [self.tok.eos_token_id])
        return out

    @torch.no_grad()
    def score(self, prefix: str, cands: Sequence[str], chunk: int = 32) -> np.ndarray:
        """Sum of log-probs of each candidate continuation after `prefix`.

        One prefix forward; candidates are scored in chunks against a copy of the prefix
        KV cache (bounded memory for large candidate sets)."""
        import copy
        p_ids = self.tok(prefix, add_special_tokens=False).input_ids
        cids = self._cand_ids(p_ids, prefix, cands)
        inp = torch.tensor([p_ids], device=self.device)
        out = self.model(input_ids=inp, use_cache=True)
        first = F.log_softmax(out.logits[0, -1].float(), -1)
        res = np.array([first[c[0]].item() for c in cids], dtype=np.float64)
        P = len(p_ids)
        chunk = self._safe_chunk(chunk, P, max(len(c) for c in cids))
        for s0 in range(0, len(cids), chunk):
            sub = cids[s0:s0 + chunk]
            L = max(len(c) for c in sub)
            if L == 1:
                continue
            C = len(sub)
            x = torch.full((C, L - 1), self.tok.pad_token_id, dtype=torch.long, device=self.device)
            att = torch.zeros((C, P + L - 1), dtype=torch.long, device=self.device)
            att[:, :P] = 1
            for i, c in enumerate(sub):
                if len(c) > 1:
                    x[i, :len(c) - 1] = torch.tensor(c[:-1], device=self.device)
                    att[i, P:P + len(c) - 1] = 1
            pos = torch.arange(P, P + L - 1, device=self.device).unsqueeze(0).expand(C, -1)
            try:
                past = copy.deepcopy(out.past_key_values)
                past.batch_repeat_interleave(C)
                o2 = self.model(input_ids=x, attention_mask=att, position_ids=pos,
                                past_key_values=past, use_cache=False)
                lp = F.log_softmax(o2.logits.float(), -1)
            except Exception:  # cache API differs across versions: recompute without cache
                full = torch.cat([inp.expand(C, -1), x], 1)
                o2 = self.model(input_ids=full, attention_mask=att)
                lp = F.log_softmax(o2.logits[:, P:].float(), -1)
            for i, c in enumerate(sub):
                for t in range(1, len(c)):
                    res[s0 + i] += lp[i, t - 1, c[t]].item()
        return res

    # ------------------------------------------------------------------ L1 contrastive decoding
    @torch.no_grad()
    def generate_contrastive(self, real: List[str], cf: List[str], alpha: float = 1.0,
                             beta: float = 0.1, max_new_tokens: int = 160, gate: str = "result",
                             stop: str = "Answer:", batch_size: int = 4,
                             space: str = "prob") -> List[str]:
        """Greedy decoding with score (1+a)*p(.|real) - a*p(.|cf)  (space='prob', default)
        or (1+a)*log p(.|real) - a*log p(.|cf)  (space='log', CAD-style).

        gate='result': contrast only while emitting a hop result (text after '=>' on the
        current line); gate='all': contrast every token (CAD-style).
        Plausibility constraint: only tokens with p_real >= beta * max p_real are eligible.
        """
        outs = []
        for i in range(0, len(real), batch_size):
            outs.extend(self._contrastive_batch(real[i:i + batch_size], cf[i:i + batch_size],
                                                alpha, beta, max_new_tokens, gate, stop, space))
        return outs

    def _contrastive_batch(self, real, cf, alpha, beta, max_new_tokens, gate, stop, space):
        B = len(real)
        enc = self._enc(list(real) + list(cf))
        att = enc.attention_mask
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        out = self.model(input_ids=enc.input_ids, attention_mask=att, position_ids=pos,
                         use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
        last_pos = pos[:, -1]
        gen = [[] for _ in range(B)]
        text = [""] * B
        done = [False] * B
        log_beta = math.log(beta)
        eos = {self.tok.eos_token_id, self.tok.pad_token_id}
        for _ in range(max_new_tokens):
            lp = F.log_softmax(logits.float(), -1)
            nxt = []
            for b in range(B):
                if done[b]:
                    nxt.append(self.tok.pad_token_id)
                    continue
                lr, lc = lp[b], lp[B + b]
                cur_line = text[b].rsplit("\n", 1)[-1]
                on = gate == "all" or (gate == "result" and "=>" in cur_line)
                if on:
                    if space == "prob":
                        s = (1 + alpha) * lr.exp() - alpha * lc.exp()
                    else:
                        s = (1 + alpha) * lr - alpha * lc
                    s[lr < lr.max() + log_beta] = -float("inf")
                    t = int(s.argmax())
                else:
                    t = int(lr.argmax())
                gen[b].append(t)
                text[b] = self.tok.decode(gen[b], skip_special_tokens=True)
                if t in eos or stop in text[b]:
                    done[b] = True
                nxt.append(t)
            if all(done):
                break
            new = torch.tensor(nxt + nxt, device=self.device).unsqueeze(1)
            att = torch.cat([att, torch.ones((2 * B, 1), dtype=att.dtype, device=self.device)], 1)
            last_pos = last_pos + 1
            out = self.model(input_ids=new, attention_mask=att, position_ids=last_pos.unsqueeze(1),
                             past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1]
        return [t.split(stop)[0] for t in text]
