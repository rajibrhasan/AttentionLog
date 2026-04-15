import numpy as np
from tqdm import tqdm

from .utils import process_attn, process_attn_prefill, calc_attn_score


class AttentionDetector():
    def __init__(self, model, pos_examples=None, neg_examples=None, use_token="first",
                 instruction="Say xxxxxx", threshold=0.5, flip=False, mode="generate"):
        self.name = "attention"
        self.attn_func = "normalize_sum"
        self.model = model
        self.important_heads = model.important_heads
        self.instruction = instruction
        self.use_token = use_token
        self.threshold = threshold
        self.flip = flip
        self.mode = mode

        if pos_examples and neg_examples:
            pos_scores, neg_scores = [], []
            for prompt in tqdm(pos_examples, desc="pos_examples"):
                pos_scores.append(self._score_sample(prompt))

            for prompt in tqdm(neg_examples, desc="neg_examples"):
                neg_scores.append(self._score_sample(prompt))

            self.threshold = (np.mean(pos_scores) + np.mean(neg_scores)) / 2
            if np.mean(neg_scores) < np.mean(pos_scores):
                self.flip = False
            else:
                self.flip = True
            print(f"Calibration: pos_mean={np.mean(pos_scores):.4f}, neg_mean={np.mean(neg_scores):.4f}, threshold={self.threshold:.4f}, flip={self.flip}")

        if pos_examples and not neg_examples:
            pos_scores = []
            for prompt in tqdm(pos_examples, desc="pos_examples"):
                pos_scores.append(self._score_sample(prompt))
            self.threshold = np.mean(pos_scores) - 4 * np.std(pos_scores)

    def _score_sample(self, prompt):
        """Score a single sample using current mode."""
        if self.mode == "prefill":
            attention_maps, input_range = self.model.prefill_inference(self.instruction, prompt)
            return self.prefill_attn2score(attention_maps, input_range)
        else:
            _, _, attention_maps, _, input_range, _ = self.model.inference(
                self.instruction, prompt, max_output_tokens=1)
            return self.attn2score(attention_maps, input_range)

    def attn2score(self, attention_maps, input_range):
        if self.use_token == "first":
            attention_maps = [attention_maps[0]]

        scores = []
        for attention_map in attention_maps:
            heatmap = process_attn(
                attention_map, input_range, self.attn_func)
            score = calc_attn_score(heatmap, self.important_heads)
            scores.append(score)

        return sum(scores) if len(scores) > 0 else 0

    def prefill_attn2score(self, attention_maps, input_range):
        heatmap = process_attn_prefill(attention_maps, input_range)
        score = calc_attn_score(heatmap, self.important_heads)
        return score

    def detect(self, data_prompt):
        focus_score = self._score_sample(data_prompt)
        if self.flip:
            detected = bool(focus_score >= self.threshold)
        else:
            detected = bool(focus_score <= self.threshold)
        return detected, {"focus_score": focus_score}
