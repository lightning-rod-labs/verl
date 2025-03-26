# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from dataclasses import dataclass


@dataclass
class ScoredItem:
    prompt_str: str
    response_str: str
    ground_truth: str
    data_source: str
    prediction: float
    brier_score: float
    resolution: float
    valid_response_length: int


class WinnerPredictionRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def _score_item(self, data_item):
        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

        data_source = data_item.non_tensor_batch['data_source']

        extra_info = data_item.non_tensor_batch.get('extra_info', None)

        prediction = self.compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        resolution = ground_truth['resolution']

        brier_score = (resolution - prediction) ** 2

        return ScoredItem(
            prompt_str=prompt_str,
            response_str=response_str,
            ground_truth=ground_truth,
            data_source=data_source,
            prediction=prediction,
            brier_score=brier_score,
            resolution=resolution,
            valid_response_length=valid_response_length,
        )

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            scored_item = self._score_item(data_item)

            scores.append(scored_item.score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        scored_items = []   
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            scored_item = self._score_item(data_item)
            scored_items.append(scored_item)
        prompt_str_to_min_brier = {}
        for scored_item in scored_items:
            if scored_item.prompt_str not in prompt_str_to_min_brier:
                prompt_str_to_min_brier[scored_item.prompt_str] = scored_item.brier_score
            else:
                prompt_str_to_min_brier[scored_item.prompt_str] = min(
                    prompt_str_to_min_brier[scored_item.prompt_str],
                    scored_item.brier_score
                )

        for i in range(len(scored_items)):
            scored_item = scored_items[i]   

            min_brier_for_prompt = prompt_str_to_min_brier[scored_item.prompt_str]
            sample_reward = 0.0
            if scored_item.brier_score == min_brier_for_prompt:
                if scored_item.resolution not in [0.0, 1.0]:
                    raise ValueError(f"Resolution must be 0 or 1, got {scored_item.resolution}")
                sample_reward = scored_item.prediction if scored_item.resolution == 0.0 else 1 - scored_item.prediction

            reward_tensor[i, scored_item.valid_response_length - 1] = sample_reward

            if scored_item.data_source not in already_print_data_sources:
                already_print_data_sources[scored_item.data_source] = 0

            if already_print_data_sources[scored_item.data_source] < self.num_examine:
                already_print_data_sources[scored_item.data_source] += 1
                print("[prompt]", scored_item.prompt_str)
                print("[response]", scored_item.response_str)
                print("[ground_truth]", scored_item.ground_truth)
                print("[score]", scored_item.score)

        return reward_tensor
