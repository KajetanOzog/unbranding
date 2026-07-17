def build_output_record(sample, model_answer):
    record = {
        "question": sample["question"],
        "model_answer": model_answer,
    }

    if "answer" in sample:
        record["reference_answer"] = sample["answer"]

    if "answers" in sample:
        record["reference_answers"] = sample["answers"]

    if "expected_brands" in sample:
        record["expected_brands"] = sample["expected_brands"]

    if "prompt_category" in sample:
        record["prompt_category"] = sample["prompt_category"]

    if "brand_category" in sample:
        record["brand_category"] = sample["brand_category"]

    return record