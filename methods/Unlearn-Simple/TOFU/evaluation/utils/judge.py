from vllm import LLM, SamplingParams


class Judge:

    def __init__(self, model_path, max_tokens=256, seed=42):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            seed=seed,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            seed=seed,
        )

    def evaluate_batch(self, prompts,):
        outputs = self.llm.generate(
            prompts,
            self.sampling_params,
            use_tqdm=True,
        )
        return [
            output.outputs[0].text.strip()
            for output in outputs
        ]

    def evaluate(self, prompt):
        return self.evaluate_batch([prompt])[0]