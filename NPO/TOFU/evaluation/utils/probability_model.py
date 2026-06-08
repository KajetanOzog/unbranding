from vllm import (
    LLM,
    SamplingParams,
)

from transformers import (
    AutoTokenizer,
)

import gc
import torch

class ProbabilityModel:

    def __init__(
        self,
        model_path,
        seed=42,
    ):

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            seed=seed,
        )

        self.tokenizer = self.llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
            seed=seed,
        )

    def score(
        self,
        question,
        reference_answer,
    ):

        full_text = (
            question.rstrip()
            + "\n"
            + reference_answer.lstrip()
        )

        question_ids = self.tokenizer(
            question.rstrip() + "\n",
            add_special_tokens=False,
        ).input_ids

        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
        ).input_ids

        answer_start = len(
            question_ids
        )

        outputs = self.llm.generate(
            [full_text],
            self.sampling_params,
            use_tqdm=False,
        )

        prompt_logprobs = (
            outputs[0].prompt_logprobs
        )

        answer_logprobs = []

        for i in range(
            answer_start,
            len(full_ids),
        ):

            token_dict = (
                prompt_logprobs[i]
            )

            if token_dict is None:
                continue

            token_id = (
                full_ids[i]
            )

            if token_id not in token_dict:
                continue

            answer_logprobs.append(
                token_dict[
                    token_id
                ].logprob
            )

        return (
            sum(answer_logprobs)
            / len(answer_logprobs)
        )


    def evaluate_batch(
        self,
        questions,
        reference_answers,
    ):

        scores = []

        for question, reference_answer in zip(
            questions,
            reference_answers,
        ):

            score = self.score(
                question,
                reference_answer,
            )

            scores.append(
                score
            )

        return scores

    def evaluate(
        self,
        question,
        reference_answer,
    ):

        return self.evaluate_batch(
            [question],
            [reference_answer],
        )[0]    
    
    def unload(self):

        del self.llm

        gc.collect()

        torch.cuda.empty_cache()