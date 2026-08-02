from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from common import TASKS


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    type: Literal["boolean", "string", "string_list", "integer", "number"]


class EvaluatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    user_prompt: str
    output: OutputConfig


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluators: dict[str, EvaluatorConfig]
    tasks: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_references(self):
        unknown_tasks = set(self.tasks) - set(TASKS)
        if unknown_tasks:
            raise ValueError(f"unknown tasks: {', '.join(sorted(unknown_tasks))}")

        missing_tasks = set(TASKS) - set(self.tasks)
        if missing_tasks:
            raise ValueError(f"missing tasks: {', '.join(sorted(missing_tasks))}")

        referenced = {
            evaluator
            for evaluators in self.tasks.values()
            for evaluator in evaluators
        }
        missing_evaluators = referenced - set(self.evaluators)
        if missing_evaluators:
            raise ValueError(
                f"unknown evaluators: {', '.join(sorted(missing_evaluators))}"
            )
        return self
