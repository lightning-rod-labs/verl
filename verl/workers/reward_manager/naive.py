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

from typing import Optional
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from dataclasses import dataclass

@dataclass
class ParsedSample:
    data_source: str
    prompt_str: str
    response_str: str
    ground_truth: dict
    extra_info: dict
    valid_response_length: int

class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def verify(self, data):
        scores = []
        parsed_samples = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

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

            parsed_samples.append(ParsedSample(
                data_source=data_source,
                prompt_str=prompt_str,
                response_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                valid_response_length=valid_response_length,
            ))


        # Pass all samples at once to be computed
        scores = self.compute_score([
            (
                parsed_sample.data_source,
                parsed_sample.response_str,
                parsed_sample.ground_truth,
                parsed_sample.extra_info,
            ) for parsed_sample in parsed_samples])
        
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, save_generations_file_name: Optional[str] = None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)


        parsed_samples = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

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

            parsed_samples.append(ParsedSample(
                data_source=data_source,
                prompt_str=prompt_str,
                response_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                valid_response_length=valid_response_length,
            ))


        print("Getting compute scores for all samples at once.")
        print(len(parsed_samples))
        # Pass all samples at once to be computed
        scores = self.compute_score([
            (
                parsed_sample.data_source,
                parsed_sample.response_str,
                parsed_sample.ground_truth,
                parsed_sample.extra_info,
            ) for parsed_sample in parsed_samples])

        for i in range(len(data)):
            parsed_sample = parsed_samples[i]
            score = scores[i]

            reward_tensor[i, parsed_sample.valid_response_length - 1] = score

        if save_generations_file_name:
            import os
            import csv
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_generations_file_name), exist_ok=True)
            
            # Open file in append mode
            with open(save_generations_file_name, 'a', newline='') as f:
                writer = csv.writer(f)
                for i, parsed_sample in enumerate(parsed_samples):
                    score = scores[i]
                    writer.writerow([
                        parsed_sample.prompt_str,
                        parsed_sample.response_str,
                        parsed_sample.ground_truth['question_id'],
                        score
                    ])

        return reward_tensor
