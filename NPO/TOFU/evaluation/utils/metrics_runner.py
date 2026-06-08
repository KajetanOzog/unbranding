from pathlib import Path

from metrics.rouge import (
    evaluate_rouge,
)

from metrics.trade_dress import (
    evaluate_trade_dress,
)

from metrics.brand_mention_judge import (
    evaluate_brand_mention_judge,
)

from metrics.world_facts import (
    evaluate_world_facts,
)

from metrics.judge_quality import (
    evaluate_judge_quality,
)

from metrics.domain_retain import (
    evaluate_domain_retain,
)


def run_metrics(
    answers_dir,
    metrics_list,
    brand_cfg,
    judge,
):

    results = {}

    answers_dir = Path(
        answers_dir
    )

    for answers_file in answers_dir.glob(
        "*.jsonl"
    ):

        dataset_name = (
            answers_file.stem
        )

        # ==========================================================
        # FORGET
        # ==========================================================

        if dataset_name == "forget":

            if "rouge" in metrics_list:

                rouge_result = (
                    evaluate_rouge(
                        answers_file
                    )
                )

                results[
                    "forget_rouge"
                ] = rouge_result[
                    "rouge_l"
                ]

            if (
                "trade_dress"
                in metrics_list
            ):

                trade_result = (
                    evaluate_trade_dress(
                        answers_file,
                        brand_cfg,
                        judge,
                    )
                )

                results[
                    "forget_trade_dress"
                ] = trade_result[
                    "trade_dress_score"
                ]

            if (
                "brand_mention_judge"
                in metrics_list
            ):

                mention_result = (
                    evaluate_brand_mention_judge(
                        answers_file,
                        brand_cfg,
                        judge,
                    )
                )

                results[
                    "forget_brand_mention_judge"
                ] = mention_result[
                    "brand_mention_score"
                ]

        # ==========================================================
        # BRAND PROMPTS
        # ==========================================================

        elif dataset_name == "brand_prompts":

            if (
                "trade_dress"
                in metrics_list
            ):

                trade_result = (
                    evaluate_trade_dress(
                        answers_file,
                        brand_cfg,
                        judge,
                    )
                )

                results[
                    "brand_prompts_trade_dress"
                ] = trade_result[
                    "trade_dress_score"
                ]

            if (
                "brand_mention_judge"
                in metrics_list
            ):

                mention_result = (
                    evaluate_brand_mention_judge(
                        answers_file,
                        brand_cfg,
                        judge,
                    )
                )

                results[
                    "brand_prompts_brand_mention_judge"
                ] = mention_result[
                    "brand_mention_score"
                ]

        # ==========================================================
        # UNMATCHED PROMPTS
        # ==========================================================

        elif dataset_name == "unmatched_prompts":

            if (
                "trade_dress"
                in metrics_list
            ):

                trade_result = (
                    evaluate_trade_dress(
                        answers_file,
                        brand_cfg,
                        judge,
                    )
                )

                results[
                    "unmatched_prompts_trade_dress"
                ] = trade_result[
                    "trade_dress_score"
                ]

            if (
                "brand_mention_judge"
                in metrics_list
            ):

                mention_result = (
                    evaluate_brand_mention_judge(
                        answers_file,
                        brand_cfg,
                        judge,
                    )
                )

                results[
                    "unmatched_prompts_brand_mention_judge"
                ] = mention_result[
                    "brand_mention_score"
                ]

        # ==========================================================
        # DOMAIN RETAIN
        # ==========================================================

        elif dataset_name == "domain_retain":

            if (
                "domain_retain"
                in metrics_list
            ):

                retain_result = (
                    evaluate_domain_retain(
                        answers_file
                    )
                )

                results[
                    "domain_retain_rouge"
                ] = retain_result[
                    "domain_retain"
                ]

            if (
                "judge_quality"
                in metrics_list
            ):

                quality_result = (
                    evaluate_judge_quality(
                        answers_file,
                        judge,
                    )
                )

                results[
                    "domain_retain_judge_quality"
                ] = quality_result[
                    "judge_quality"
                ]

        # ==========================================================
        # WORLD FACTS
        # ==========================================================

        elif dataset_name == "world_facts":

            if (
                "world_facts"
                in metrics_list
            ):

                facts_result = (
                    evaluate_world_facts(
                        answers_file
                    )
                )

                results[
                    "world_facts_accuracy"
                ] = facts_result[
                    "world_facts_accuracy"
                ]

        # ==========================================================
        # ALPACA
        # ==========================================================

        elif dataset_name == "alpaca":

            if (
                "rouge"
                in metrics_list
            ):

                rouge_result = (
                    evaluate_rouge(
                        answers_file
                    )
                )

                results[
                    "alpaca_rouge"
                ] = rouge_result[
                    "rouge_l"
                ]

            if (
                "judge_quality"
                in metrics_list
            ):

                quality_result = (
                    evaluate_judge_quality(
                        answers_file,
                        judge,
                    )
                )

                results[
                    "alpaca_judge_quality"
                ] = quality_result[
                    "judge_quality"
                ]

    return results