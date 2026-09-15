"""
AREA inference model for Qwen2.5-VL.

Architecture implements the AREA inference path with three components:
  • Combined probe pass that captures both visual (L_vis) and text (L_txt) attention.
  • EAEG + DG-LT decisions computed from those attention maps.
  • ETML: monitored token-by-token generation with KV-cache-compatible reminder injection.
"""
import copy
import types
from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy
import torch
from PIL import Image
from accelerate.hooks import remove_hook_from_module
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

# Local attention forwards expose the selected attention rows without retaining
# full sequence-by-sequence attention tensors.
from .attention import (
    create_qwen2_5_vl_self_attn_forward,
    qwen2_5_vl_decoderlayer_forward,
    qwen2_5_vl_layernorm_forward,
    qwen2_5_vl_mlp_forward,
    qwen2_5_vl_self_attn_forward,
)

from .algorithm import (
    compute_sentence_scores,
    compute_eaeg_text,
    compute_visual_centroid,
    compute_eaeg_vision,
    compute_step_entropy,
    is_entropy_peak,
    compute_bbox_pixels,
)
from .running_stats import OnlineMedian

# ─── Sink configuration ─────────────────────────────────────────────────────────
SINK_DIMS: Dict[str, List[int]] = {
    "Qwen/Qwen2.5-VL-7B-Instruct": [458, 2570],
    "Qwen/Qwen2.5-VL-3B-Instruct": [318, 1874, 1819],
    "Qwen/Qwen2.5-VL-32B-Instruct": [4675, 3094],
}
SINK_PERCENTILE = 25  # 25th-percentile threshold 
PASSAGE_DELIMITER = "\n\n\n"

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ─── Base class ──────────────────────────────────────────────────────────────────

class AREAInferenceModel:
    """Shared input-building and tokenisation utilities, model-agnostic."""

    def __init__(self, args):
        self.args = args
        self.token_vis_start: Optional[str] = None
        self.token_vis_end: Optional[str] = None

    # ── tokenisation helpers ───────────────────────────────────────────────────

    def find_text_token_spans(
        self,
        input_ids: List[int],
        target_text: str,
        raise_if_not_found: bool = False,
    ) -> List[Tuple[int, int]]:
        tokenizer = self.processor.tokenizer
        source = tokenizer.decode(input_ids)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        target = tokenizer.decode(target_ids)

        if target not in source:
            if raise_if_not_found:
                raise ValueError(f"Text not found: {target!r}")
            return []

        spans = []
        n_match = source.count(target)
        n_match_left = n_match
        start = 0
        while n_match_left > 0:
            idx = source.find(target, start)
            if idx == -1:
                break
            # Map character offset to token offset
            pre_tokens = tokenizer.encode(source[:idx], add_special_tokens=False)
            tok_start = len(pre_tokens)
            pre_plus_target_tokens = tokenizer.encode(source[:idx + len(target)], add_special_tokens=False)
            tok_end = len(pre_plus_target_tokens)
            if tok_end > len(input_ids):
                tok_end = -1
            spans.append((tok_start, tok_end))
            n_match_left -= 1
            if tok_end == -1:
                break
            start = idx + 1
        return spans

    def get_context_token_span(self, context: str, input_ids: List[int]) -> Tuple[int, int]:
        spans = self.find_text_token_spans(input_ids, context)
        if not spans:
            return (0, -1)
        return spans[0]

    def get_sentence_token_spans(self, context_ids: List[int]) -> Tuple[List[Tuple[int, int]], List[str]]:
        tokenizer = self.processor.tokenizer
        context_text = tokenizer.decode(context_ids)
        context_tokens_text = [tokenizer.decode([tid]).replace(" ", "") for tid in context_ids]
        nlp = _get_nlp()
        sents = [s.text for s in nlp(context_text).sents]
        # Merge very short sentences
        for i in range(len(sents)):
            if len(sents[i].strip()) <= 5:
                if i < len(sents) - 1:
                    sents[i + 1] = sents[i] + sents[i + 1]
                    sents[i] = ""
                elif i > 0:
                    sents[i - 1] = sents[i - 1] + sents[i]
                    sents[i] = ""
        sents = [s for s in sents if s]

        sent_token_spans: List[Tuple[int, int]] = []
        tk_start = 0
        for sent in sents:
            sent = sent.lstrip(" ")
            sent_text = sent.replace(" ", "")
            n_tok = len(tokenizer.encode(sent, add_special_tokens=False))
            span_text = tokenizer.decode(context_ids[tk_start:tk_start + n_tok]).replace(" ", "")
            span_inc = sent_text in span_text
            sent_inc = span_text in sent_text
            length = n_tok
            if span_inc and not sent_inc:
                while length > 0:
                    length -= 1
                    if tk_start + length >= len(context_tokens_text):
                        break
                    del_tok = context_tokens_text[tk_start + length]
                    span_text = span_text.rstrip(del_tok)
                    if sent_text not in span_text:
                        length += 1
                        break
            elif not span_inc:
                while True:
                    if tk_start + length >= len(context_tokens_text):
                        break
                    add_tok = context_tokens_text[tk_start + length]
                    length += 1
                    span_text = span_text + add_tok
                    if sent_text in span_text:
                        break
            tk_end = tk_start + length
            sent_token_spans.append((tk_start, tk_end))
            tk_start = tk_end
            if not span_text.endswith(sent_text):
                tk_start -= 1

        assert len(sent_token_spans) == len(sents)
        return sent_token_spans, sents


# ─── Qwen2.5-VL AREA model ─────────────────────────────────────────────────

class AREAInferenceModelQwen2_5_VL(AREAInferenceModel):
    """
    Implements AREA for Qwen/Qwen2.5-VL-*-Instruct models.

    Loads two model instances:
      self.model       — flash_attention_2, used only for pure-flash generation
                         (when ETML is disabled with K_max=0 and gates are both ON).
      self.model_probe — eagerly patched layers for probe passes and ETML generation.
    """

    def __init__(self, args):
        super().__init__(args)

        self.processor = AutoProcessor.from_pretrained(
            args.model_name,
            min_pixels=getattr(args, 'min_pixels', 3136),
            max_pixels=getattr(args, 'max_pixels', 301056),
            padding_side="left",
            trust_remote_code=True,
        )

        # Use flash_attention_2 if available, otherwise fall back to sdpa
        try:
            import flash_attn  # noqa: F401
            _attn_impl = "flash_attention_2"
        except ImportError:
            _attn_impl = "sdpa"

        print(f"AREA: loading flash model (attn={_attn_impl}) …")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            attn_implementation=_attn_impl,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            trust_remote_code=True,
        )
        self.model.config.use_cache = True
        self.model.eval()

        print("AREA: loading probe model (eager attention on L_probe) …")
        self.model_probe = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name,
            attn_implementation=_attn_impl,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            trust_remote_code=True,
        )
        self.model_probe.gradient_checkpointing_disable()

        # Patch L_probe = [n//4, n) layers with eager attention (union of L_vis and L_txt)
        probe_lm = self.model_probe.model.language_model
        n = len(probe_lm.layers)
        self._n_layers = n
        self._L_vis_start = n // 4
        self._L_vis_end = 3 * n // 4
        self._L_txt_start = n // 2
        self._L_txt_end = n
        self._L_probe_start = n // 4   # union start
        self._L_probe_end = n           # union end

        config_eager = copy.deepcopy(probe_lm.layers[0].self_attn.config)
        config_eager._attn_implementation = "eager"
        config_eager.output_attentions = True

        # Non-probe layers keep the same attn impl used to load the model
        # (sdpa when flash_attn is not installed; never force flash_attention_2).
        config_non_probe = copy.deepcopy(probe_lm.layers[0].self_attn.config)
        config_non_probe._attn_implementation = _attn_impl
        config_non_probe.output_attentions = False

        for i, layer in enumerate(probe_lm.layers):
            if self._L_probe_start <= i < self._L_probe_end:
                layer.self_attn.config = config_eager
                remove_hook_from_module(layer, "attn_output_hook")
                layer.forward = types.MethodType(qwen2_5_vl_decoderlayer_forward, layer)
                layer.self_attn.forward = types.MethodType(qwen2_5_vl_self_attn_forward, layer.self_attn)
                layer.input_layernorm.forward = types.MethodType(qwen2_5_vl_layernorm_forward, layer.input_layernorm)
                layer.post_attention_layernorm.forward = types.MethodType(qwen2_5_vl_layernorm_forward, layer.post_attention_layernorm)
                layer.mlp.forward = types.MethodType(qwen2_5_vl_mlp_forward, layer.mlp)
            else:
                layer.self_attn.config = config_non_probe

        for p in self.model_probe.parameters():
            p.requires_grad = False

        # Probe model will be used with use_cache=False for the probe pass,
        # and with use_cache=True for the ETML generation loop.
        self.model_probe.config.use_cache = False
        self.model_probe.model.config.use_cache = False
        self.model_probe.eval()

        self.token_vis_start = "<|vision_start|>"
        self.token_vis_end = "<|vision_end|>"

        # DG-LT running statistics (persists over the evaluation stream)
        self._median_txt = OnlineMedian()
        self._median_vis = OnlineMedian()
        self._seen = 0

        # Sink dims
        self._sink_dims = SINK_DIMS.get(args.model_name, [318, 1874, 1819])
        print(f"AREA: sink dims = {self._sink_dims}")
        print("AREA: model loaded.")

    # ── Input building ─────────────────────────────────────────────────────────

    def _build_probe_inputs(
        self,
        question: str,
        image: Image.Image,
        context: str,
    ):
        """Build inputs for the combined probe pass (no highlighting)."""
        import prompts as P
        messages = []
        if P.SYSTEM_PROMPT:
            messages.append({
                'role': 'system',
                'content': [{'type': 'text', 'text': P.SYSTEM_PROMPT + P.SELF_ELICIT_SYSTEM_PROMPT_VQA_TEXT}],
            })
        messages.append({
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': question},
            ],
        })
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
            max_length=getattr(self.args, 'model_max_length', 16384),
            truncation=True,
        )
        return inputs

    def _build_generation_inputs(
        self,
        question: str,
        image: Image.Image,
        elicited_context: str,
        image_cropped: Optional[Image.Image],
        gate_txt: bool,
        gate_vis: bool,
    ):
        """
        Build inputs for the answer generation pass.
        Wraps the highlighted content with the appropriate markers (Stage E).
        """
        import prompts as P
        system_txt = P.SYSTEM_PROMPT
        if gate_txt:
            system_txt += P.SELF_ELICIT_SYSTEM_PROMPT_VQA_TEXT
        if gate_vis and image_cropped is not None:
            system_txt += P.SELF_ELICIT_SYSTEM_PROMPT_VQA_IMG

        messages = []
        if system_txt:
            messages.append({
                'role': 'system',
                'content': [{'type': 'text', 'text': system_txt}],
            })

        user_content: List[dict] = []

        if gate_vis and image_cropped is not None:
            # Show full image first, then the cropped region with highlight markers.
            user_content.append({'type': 'image'})
            user_content.append({'type': 'text', 'text': '<START_IMPORTANT_IMG>'})
            user_content.append({'type': 'image'})
            user_content.append({'type': 'text', 'text': '<END_IMPORTANT_IMG>\n'})
            images = [image, image_cropped]
        else:
            user_content.append({'type': 'image'})
            images = [image]

        user_content.append({'type': 'text', 'text': question})
        messages.append({'role': 'user', 'content': user_content})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
            max_length=getattr(self.args, 'model_max_length', 16384),
            truncation=True,
        )
        return inputs

    # ── Probe pass: text attention + sink scores ───────────────────────────────

    @torch.inference_mode()
    def _probe_text(
        self,
        inputs,
        context_span: Tuple[int, int],
        sent_spans: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass on model_probe with hooks on L_txt (last-token→context) and
        L_sink (hidden-state sink scores).

        Returns:
            a_txt   — [N_C] per-token attention (already averaged over L_txt layers & heads)
            s_sink  — [N_V] per-visual-token sink score (averaged over L_sink layers)
        """
        lm = self.model_probe.model.language_model
        ctx_start, ctx_end = context_span

        # Find visual token indices
        input_ids_list = inputs.input_ids.cpu().tolist()[0]
        vis_start_id = self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_start)
        vis_end_id = self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_end)
        ids_tensor = inputs.input_ids[0]
        vis_start_positions = (ids_tensor == vis_start_id).nonzero(as_tuple=True)[0]
        vis_end_positions = (ids_tensor == vis_end_id).nonzero(as_tuple=True)[0]
        if len(vis_start_positions) == 0 or len(vis_end_positions) == 0:
            n_ctx = ctx_end - ctx_start if ctx_end != -1 else 0
            return np.zeros(n_ctx, dtype=np.float32), np.array([], dtype=np.float32)
        vis_start_idx = vis_start_positions[0].item()
        vis_end_idx = vis_end_positions[0].item()
        n_vis = vis_end_idx - vis_start_idx - 1

        attention_data: Dict[int, np.ndarray] = {}
        sink_data: Dict[int, np.ndarray] = {}

        def make_txt_hook(layer_idx, cs, ce):
            def hook(module, inp, out):
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                    w = out[1]  # [batch, heads, q_len, k_len]
                    # attention from last token (probe token) to context tokens
                    relevant = w[0, :, -1, cs:ce]  # [heads, N_C]
                    attention_data[layer_idx] = relevant.mean(dim=0).detach().cpu().float().numpy()
            return hook

        def make_sink_hook(layer_idx, vs, ve, sdims):
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                vis_hs = hs[0, vs + 1:ve, :]  # [N_V, d]
                sink_vals = vis_hs[:, sdims].abs().max(dim=1).values
                rms = torch.sqrt(vis_hs.pow(2).mean(dim=1))
                sink_data[layer_idx] = (sink_vals / (rms + 1e-6)).detach().cpu().float().numpy()
            return hook

        hooks = []
        sink_dims_t = torch.tensor(self._sink_dims, dtype=torch.long)
        for i, layer in enumerate(lm.layers):
            if self._L_txt_start <= i < self._L_txt_end:
                h = layer.self_attn.register_forward_hook(
                    make_txt_hook(i, ctx_start, ctx_end if ctx_end != -1 else inputs.input_ids.shape[1])
                )
                hooks.append(h)
            if self._L_vis_start <= i < self._L_vis_end:
                h = layer.register_forward_hook(
                    make_sink_hook(i, vis_start_idx, vis_end_idx, sink_dims_t)
                )
                hooks.append(h)

        target_dev = self.model_probe.model.visual.patch_embed.proj.weight.device
        model_inputs = {k: v.to(target_dev) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        try:
            self.model_probe(**model_inputs, output_attentions=False, use_cache=False)
        finally:
            for h in hooks:
                h.remove()

        # Aggregate text attention
        if attention_data:
            sorted_layers = sorted(attention_data)
            arr = np.stack([attention_data[l] for l in sorted_layers], axis=0)  # [L_txt, N_C]
            arr /= (arr.sum(axis=1, keepdims=True) + 1e-9)
            a_txt = arr.mean(axis=0)
        else:
            n_ctx = (ctx_end - ctx_start) if ctx_end != -1 else 0
            a_txt = np.zeros(n_ctx, dtype=np.float32)

        # Aggregate sink scores
        if sink_data:
            sorted_sink = sorted(sink_data)
            s_sink = np.stack([sink_data[l] for l in sorted_sink], axis=0).mean(axis=0)
        else:
            s_sink = np.zeros(n_vis, dtype=np.float32)

        return a_txt.astype(np.float32), s_sink.astype(np.float32)

    # ── Probe pass: visual attention ───────────────────────────────────────────

    @torch.inference_mode()
    def _probe_visual(
        self,
        inputs,
        entity: Optional[str],
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """
        Forward pass on model_probe with entity-range attention on L_vis layers.
        Temporarily monkey-patches L_vis layers.

        Returns:
            a_vis_raw — [N_V] raw visual attention from entity tokens, or None
            grid_shape — (H_grid, W_grid) or None
        """
        input_ids_list = inputs.input_ids.cpu().tolist()[0]

        # Find entity token span
        if entity is not None:
            entity_spans = self.find_text_token_spans(input_ids_list, entity)
            if not entity_spans:
                entity_start, entity_end = -1, len(input_ids_list)
            else:
                entity_start, entity_end = entity_spans[0]
                if entity_end == -1:
                    entity_end = len(input_ids_list)
        else:
            entity_start, entity_end = -1, len(input_ids_list)

        # Find visual token positions
        ids_tensor = inputs.input_ids[0]
        vis_start_id = self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_start)
        vis_end_id = self.processor.tokenizer.convert_tokens_to_ids(self.token_vis_end)
        vs_pos = (ids_tensor == vis_start_id).nonzero(as_tuple=True)[0]
        ve_pos = (ids_tensor == vis_end_id).nonzero(as_tuple=True)[0]
        if len(vs_pos) == 0 or len(ve_pos) == 0:
            return None, None
        vis_start_idx = vs_pos[0].item()
        vis_end_idx = ve_pos[0].item()

        # Temporarily patch L_vis layers with entity-range attention
        lm = self.model_probe.model.language_model
        orig_forwards = {}
        for i in range(self._L_vis_start, self._L_vis_end):
            layer = lm.layers[i]
            orig_forwards[i] = layer.self_attn.forward
            layer.self_attn.forward = types.MethodType(
                create_qwen2_5_vl_self_attn_forward(entity_start, entity_end),
                layer.self_attn,
            )

        attn_data: Dict[int, np.ndarray] = {}

        def make_vis_hook(layer_idx, vs, ve):
            def hook(module, inp, out):
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                    w = out[1]  # [batch, heads, entity_len, seq_len]
                    # entity tokens → visual tokens
                    entity_to_vis = w[0, :, :, vs + 1:ve]  # [heads, entity_len, N_V]
                    mean_over_entity = entity_to_vis.mean(dim=1)  # [heads, N_V]
                    attn_data[layer_idx] = mean_over_entity.mean(dim=0).detach().cpu().float().numpy()
            return hook

        hooks = []
        for i in range(self._L_vis_start, self._L_vis_end):
            h = lm.layers[i].self_attn.register_forward_hook(
                make_vis_hook(i, vis_start_idx, vis_end_idx)
            )
            hooks.append(h)

        target_dev = self.model_probe.model.visual.patch_embed.proj.weight.device
        model_inputs = {k: v.to(target_dev) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        try:
            self.model_probe(**model_inputs, output_attentions=False, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
            for i, fw in orig_forwards.items():
                lm.layers[i].self_attn.forward = fw

        if not attn_data:
            return None, None

        sorted_layers = sorted(attn_data)
        arr = np.stack([attn_data[l] for l in sorted_layers], axis=0)  # [L_vis, N_V]
        a_vis_raw = arr.mean(axis=0).astype(np.float32)

        # Determine grid shape from image_grid_thw
        grid_thw = inputs.image_grid_thw[0]
        h_grid, w_grid = int(grid_thw[1]), int(grid_thw[2])
        n_vis = a_vis_raw.shape[0]
        # Handle merged tokens (Qwen2.5-VL merges 2×2 patches)
        if n_vis * 4 == h_grid * w_grid:
            grid_shape = (h_grid // 2, w_grid // 2)
        elif n_vis * 2 == h_grid * w_grid:
            grid_shape = (h_grid, w_grid // 2)
        elif n_vis == h_grid * w_grid:
            grid_shape = (h_grid, w_grid)
        else:
            side = int(np.sqrt(n_vis))
            grid_shape = (side, side)

        return a_vis_raw, grid_shape

    # ── ETML: monitored generation loop ───────────────────────────────────────

    @torch.inference_mode()
    def _generate_etml(
        self,
        gen_inputs,
        context_start_in_gen: int,
        context_end_in_gen: int,
        sent_spans: List[Tuple[int, int]],
        sents: List[str],
        k_max: int,
        K_max: int,
        kappa: float,
        max_new_tokens: int,
        eos_token_id: int,
    ) -> Tuple[str, int]:
        """
        Implements Stage F: token-by-token generation with ETML re-injection.

        Uses model_probe (eager attention enabled on L_txt layers) so that
        per-step attention weights are captured via hooks.
        KV cache is maintained and reminder injections are append-only.
        """
        # Enable KV cache for generation
        self.model_probe.config.use_cache = True
        self.model_probe.model.config.use_cache = True

        target_dev = self.model_probe.model.visual.patch_embed.proj.weight.device
        model_inputs = {k: v.to(target_dev) if isinstance(v, torch.Tensor) else v for k, v in gen_inputs.items()}

        # ── Prefill (process the initial prompt) ────────────────────────────
        with torch.no_grad():
            prefill_out = self.model_probe(
                **model_inputs,
                output_attentions=False,
                use_cache=True,
                return_dict=True,
            )
        past_kv = prefill_out.past_key_values
        next_token_logits = prefill_out.logits[0, -1, :]
        next_token_id = int(next_token_logits.argmax())

        generated_ids = [next_token_id]
        entropy_history: List[float] = []
        r = 0  # re-look counter
        # Running absolute position after the prompt
        current_abs_pos = model_inputs['input_ids'].shape[1]

        # Context window for entropy hooks (relative to the ORIGINAL gen prompt)
        ctx_s, ctx_e = context_start_in_gen, context_end_in_gen

        for step in range(1, max_new_tokens):
            if next_token_id == eos_token_id:
                break

            # ── capture per-step attention ───────────────────────────────────
            step_attn: Dict[int, np.ndarray] = {}
            lm = self.model_probe.model.language_model

            def make_step_hook(lidx):
                def hook(module, inp, out):
                    if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                        w = out[1]  # [1, heads, 1, total_kv_len]
                        # attention from the current decode position to the ORIGINAL context
                        # kv_len may exceed ctx_e because reminders were appended;
                        # we always look at the ORIGINAL context window.
                        kv_len = w.shape[-1]
                        safe_ctx_e = min(ctx_e, kv_len)
                        if safe_ctx_e > ctx_s:
                            relevant = w[0, :, -1, ctx_s:safe_ctx_e]  # [heads, N_C]
                            step_attn[lidx] = relevant.mean(dim=0).detach().cpu().float().numpy()
                return hook

            hooks = []
            for i in range(self._L_txt_start, self._L_txt_end):
                h = lm.layers[i].self_attn.register_forward_hook(make_step_hook(i))
                hooks.append(h)

            # ── one decode step ──────────────────────────────────────────────
            decode_input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=target_dev)
            cache_position = torch.tensor([current_abs_pos], dtype=torch.long, device=target_dev)
            try:
                with torch.no_grad():
                    step_out = self.model_probe(
                        input_ids=decode_input_ids,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_attentions=False,
                        cache_position=cache_position,
                        return_dict=True,
                    )
            finally:
                for h in hooks:
                    h.remove()

            past_kv = step_out.past_key_values
            current_abs_pos += 1
            next_token_logits = step_out.logits[0, -1, :]
            candidate_next_token_id = int(next_token_logits.argmax())

            # ── compute entropy from captured attention ───────────────────────
            if step_attn:
                sorted_l = sorted(step_attn)
                attn_arr = np.stack([step_attn[l] for l in sorted_l], axis=0)
                attn_arr /= (attn_arr.sum(axis=1, keepdims=True) + 1e-9)
                a_step = attn_arr.mean(axis=0)
                E_t = compute_step_entropy(a_step)
            else:
                E_t = 0.0
                a_step = np.array([], dtype=np.float32)

            # ── ETML re-injection ────────────────────────────────────────────
            if (
                r < K_max
                and len(a_step) > 0
                and is_entropy_peak(E_t, entropy_history, kappa)
            ):
                # Re-select evidence with EAEG anchored at current decode position
                sent_scores_new = compute_sentence_scores(a_step, sent_spans)
                k_new, _ = compute_eaeg_text(sent_scores_new, k_max)
                top_idx = np.argsort(sent_scores_new)[::-1][:k_new].tolist()
                reminder_sents = [sents[idx] for idx in sorted(top_idx)]
                reminder_text = (
                    "<START_IMPORTANT_TXT> "
                    + " ".join(reminder_sents)
                    + " <END_IMPORTANT_TXT>"
                )
                reminder_ids = self.processor.tokenizer.encode(
                    reminder_text, add_special_tokens=False
                )
                reminder_tensor = torch.tensor(
                    [reminder_ids], dtype=torch.long, device=target_dev
                )
                # Extend KV cache with reminder tokens (append-only, no invalidation).
                # The reminder changes the prefix for the next generated token, so the
                # next-token decision must be recomputed from the reminder tail logits.
                # Using the pre-reminder logits after appending the reminder places a
                # token selected under the old prefix after a different prefix.
                with torch.no_grad():
                    reminder_out = self.model_probe(
                        input_ids=reminder_tensor,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_attentions=False,
                        return_dict=True,
                    )
                past_kv = reminder_out.past_key_values
                next_token_logits = reminder_out.logits[0, -1, :]
                next_token_id = int(next_token_logits.argmax())
                current_abs_pos += len(reminder_ids)
                r += 1
            else:
                next_token_id = candidate_next_token_id

            entropy_history.append(E_t)
            generated_ids.append(next_token_id)

        # Restore probe model to no-cache mode (used for forward passes elsewhere)
        self.model_probe.config.use_cache = False
        self.model_probe.model.config.use_cache = False

        output_text = self.processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text, r

    # ── Main AREA pipeline ──────────────────────────────────────────────────

    @torch.inference_mode()
    def area_generate(
        self,
        query: str,
        image: Image.Image,
        context: str,
        entity: Optional[str],
        k_max: int = 3,
        beta_min: float = 1.5,
        beta_max: float = 2.5,
        kappa: float = 1.0,
        K_max: int = 4,
        warmup: int = 50,
        eps: float = 1e-8,
        max_new_tokens: int = 128,
    ) -> Tuple[str, dict]:
        """
        Full AREA pipeline: Stage A → Stage F.

        Returns (answer_text, info_dict) where info_dict holds diagnostics.
        """
        import prompts as P
        info: dict = {}

        has_context = bool(context and context.strip())

        # ── Stage A+B: combined probe pass ────────────────────────────────────
        if has_context:
            question_for_probe = P.CONTEXT_VQA_PROMPT_SELF_ELICIT.format(
                question=query, context=context
            )
        else:
            question_for_probe = query

        probe_inputs = self._build_probe_inputs(question_for_probe, image, context)
        input_ids_list = probe_inputs.input_ids.cpu().tolist()[0]

        # Locate context span in the tokenised probe input
        if has_context:
            ctx_span = self.get_context_token_span(context, input_ids_list)
        else:
            ctx_span = (0, 0)
        ctx_start, ctx_end = ctx_span
        if ctx_end == -1:
            ctx_end = len(input_ids_list)

        context_ids = input_ids_list[ctx_start:ctx_end]
        sent_spans, sents = self.get_sentence_token_spans(context_ids)

        # Text probe: a_txt [N_C], s_sink [N_V]
        a_txt, s_sink = self._probe_text(probe_inputs, (ctx_start, ctx_end), sent_spans)

        # Visual probe: a_vis_raw [N_V], grid_shape
        a_vis_raw, grid_shape = self._probe_visual(probe_inputs, entity)

        # ── Stage B continued: sink filtering + reference preprocessing ──────
        if a_vis_raw is not None and len(s_sink) == len(a_vis_raw):
            tau = np.percentile(s_sink, SINK_PERCENTILE)
            sink_mask = s_sink >= tau
            a_vis = a_vis_raw.copy()
            a_vis[sink_mask] = 0.0
        else:
            a_vis = a_vis_raw if a_vis_raw is not None else np.array([], dtype=np.float32)
            sink_mask = np.zeros(len(a_vis), dtype=bool)

        # 2-pixel border masking (removes edge artefacts; matches reference preprocessing)
        if len(a_vis) > 0 and grid_shape is not None:
            h_g, w_g = grid_shape
            if h_g > 4 and w_g > 4:
                a_vis_2d = a_vis.reshape(grid_shape)
                a_vis_2d[0:2, :] = 0.0
                a_vis_2d[-2:, :] = 0.0
                a_vis_2d[:, 0:2] = 0.0
                a_vis_2d[:, -2:] = 0.0
                a_vis = a_vis_2d.ravel()

        # [0,1] min-max normalisation (sharpens contrast before centroid; matches LoT)
        if len(a_vis) > 0:
            a_min, a_max = float(a_vis.min()), float(a_vis.max())
            if a_max > a_min + eps:
                a_vis = (a_vis - a_min) / (a_max - a_min)
            else:
                a_vis = np.zeros_like(a_vis)

        info['n_vis_tokens'] = len(a_vis)
        info['n_sink'] = int(sink_mask.sum()) if len(sink_mask) > 0 else 0

        # ── Stage C: EAEG ─────────────────────────────────────────────────────
        sent_scores = compute_sentence_scores(a_txt, sent_spans) if has_context else np.array([])
        k_txt, H_txt = compute_eaeg_text(sent_scores, k_max) if has_context else (0, 0.0)
        info['k_txt'] = k_txt
        info['H_txt'] = H_txt

        beta = 2.0  # fixed default (overridden if visual probe succeeded)
        H_vis = 0.0
        c_x = c_y = sigma_x = sigma_y = 0.0
        M_vis_tilde = None
        if len(a_vis) > 0 and grid_shape is not None:
            c_x, c_y, sigma_x, sigma_y, M_vis_tilde = compute_visual_centroid(a_vis, grid_shape)
            if any(np.isnan(v) for v in (c_x, c_y, sigma_x, sigma_y)):
                a_vis = np.array([], dtype=np.float32)
                grid_shape = None
            else:
                beta, H_vis = compute_eaeg_vision(M_vis_tilde, sigma_x, sigma_y, beta_min, beta_max)
        info['beta'] = beta
        info['H_vis'] = H_vis

        # Tentative grid-space bbox (computed before gate decision; needed for δ_vis)
        grid_bbox = None
        if len(a_vis) > 0 and grid_shape is not None:
            h_g, w_g = grid_shape
            x1g = max(0, int(c_x - beta * sigma_x))
            y1g = max(0, int(c_y - beta * sigma_y))
            x2g = min(w_g, int(c_x + beta * sigma_x))
            y2g = min(h_g, int(c_y + beta * sigma_y))
            if x2g > x1g and y2g > y1g:
                grid_bbox = (x1g, y1g, x2g, y2g)

        # ── Stage D: DG-LT ────────────────────────────────────────────────────
        # δ_txt: surprisal of text-evidence attention (context-dependent)
        delta_txt = 0.0
        if has_context and k_txt > 0 and len(sent_scores) > 0:
            top_idx = np.argsort(sent_scores)[::-1][:k_txt].tolist()
            top_spans = [sent_spans[i] for i in top_idx]
            ctx_total = float(a_txt.sum()) + eps
            ev_mass = sum(float(a_txt[s:e].sum()) for s, e in top_spans if e > s)
            pi_txt = ev_mass / ctx_total
            delta_txt = float(-np.log2(pi_txt + eps))
        else:
            top_spans = []

        # δ_vis: bbox concentration — context-independent; high = crop spatially focused
        delta_vis = 0.0
        if grid_bbox is not None and len(a_vis) > 0 and grid_shape is not None:
            x1g, y1g, x2g, y2g = grid_bbox
            a_vis_2d = a_vis.reshape(grid_shape)
            vis_total = float(a_vis.sum()) + eps
            delta_vis = float(a_vis_2d[y1g:y2g, x1g:x2g].sum()) / vis_total

        median_txt = self._median_txt.get()
        median_vis = self._median_vis.get()

        gate_txt = has_context and k_txt > 0 and (
            self._seen < warmup or delta_txt >= median_txt
        )
        gate_vis = self._seen < warmup or delta_vis >= median_vis

        # Update running medians and counter
        if has_context:
            self._median_txt.push(delta_txt)
        self._median_vis.push(delta_vis)
        self._seen += 1

        info['delta_txt'] = delta_txt
        info['delta_vis'] = delta_vis
        info['gate_txt'] = gate_txt
        info['gate_vis'] = gate_vis

        # ── Stage E: build highlighted context and image ───────────────────────
        if gate_txt and has_context and len(sent_scores) > 0:
            # Sort selected sentence indices in original order for natural reading
            selected_idx = sorted(np.argsort(sent_scores)[::-1][:k_txt].tolist())
            selected_sents = set(selected_idx)
            parts = []
            for i, sent in enumerate(sents):
                is_valid = len(sent.replace(" ", "")) > 5 and sent.strip() != '# Wiki Article:'
                if i in selected_sents and is_valid:
                    parts.append(f"<START_IMPORTANT_TXT> {sent} <END_IMPORTANT_TXT>")
                else:
                    parts.append(sent)
            elicited_context = " ".join(parts)
        else:
            elicited_context = context

        if gate_vis and a_vis is not None and len(a_vis) > 0 and grid_shape is not None:
            bbox_px = compute_bbox_pixels(
                c_x, c_y, sigma_x, sigma_y, beta, grid_shape, image.size
            )
            x1, y1, x2, y2 = bbox_px
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                image_cropped = image.crop(bbox_px)
            else:
                image_cropped = None
        else:
            image_cropped = None

        info['image_cropped'] = image_cropped is not None

        # ── Stage E continued: build question with highlighted context ─────────
        if has_context:
            highlighted_question = P.CONTEXT_VQA_PROMPT_training.format(
                question=query, context=elicited_context
            )
        else:
            highlighted_question = query

        # ── Stage F: ETML generation ──────────────────────────────────────────
        gen_inputs = self._build_generation_inputs(
            highlighted_question, image, elicited_context, image_cropped, gate_txt, gate_vis
        )

        eos_token_id = self.processor.tokenizer.eos_token_id or 151645  # Qwen EOS

        if K_max > 0 and has_context:
            # Token-by-token ETML loop (uses model_probe)
            gen_input_ids_list = gen_inputs.input_ids.cpu().tolist()[0]
            if has_context:
                gen_ctx_span = self.get_context_token_span(elicited_context, gen_input_ids_list)
                gen_ctx_start = gen_ctx_span[0]
                gen_ctx_end = gen_ctx_span[1]
                if gen_ctx_end == -1:
                    gen_ctx_end = len(gen_input_ids_list)
            else:
                gen_ctx_start = gen_ctx_end = 0

            answer, n_reinjections = self._generate_etml(
                gen_inputs,
                context_start_in_gen=gen_ctx_start,
                context_end_in_gen=gen_ctx_end,
                sent_spans=sent_spans,
                sents=sents,
                k_max=k_max,
                K_max=K_max,
                kappa=kappa,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
            info['etml_reinjections'] = n_reinjections
        else:
            # Fast generation (no ETML monitoring) using model_probe or self.model
            target_dev = self.model.model.visual.patch_embed.proj.weight.device
            model_inputs = {k: v.to(target_dev) if isinstance(v, torch.Tensor) else v for k, v in gen_inputs.items()}
            in_len = model_inputs['input_ids'].shape[1]
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            out_ids = generated_ids[0][in_len:]
            answer = self.processor.tokenizer.decode(
                out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            info['etml_reinjections'] = 0

        return answer, info
    # ── Convenience generation interface ───────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        question: str,
        image_query: Image.Image,
        context: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> Tuple[str, int, int, int]:
        """Thin wrapper for call sites that only need (answer, in_len, out_len, n_image_tokens)."""
        import prompts as P
        k_max = getattr(self.args, 'k_max', 3)
        beta_min = getattr(self.args, 'beta_min', 1.5)
        beta_max = getattr(self.args, 'beta_max', 2.5)
        kappa = getattr(self.args, 'kappa', 1.0)
        K_max = getattr(self.args, 'K_max', 4)
        warmup = getattr(self.args, 'warmup', 50)
        max_new_tokens = getattr(self.args, 'max_new_tokens', 128)

        answer, info = self.area_generate(
            query=question,
            image=image_query,
            context=context or "",
            entity=entity,
            k_max=k_max,
            beta_min=beta_min,
            beta_max=beta_max,
            kappa=kappa,
            K_max=K_max,
            warmup=warmup,
            max_new_tokens=max_new_tokens,
        )
        # Return dummy token counts for output-format compatibility
        return answer, 0, 0, 0
