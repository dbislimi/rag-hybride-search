from pydantic import BaseModel, Field


class Evaluation(BaseModel):
    date: str
    metrics: dict[str, float]


class MethodEvals(BaseModel):
    parameters: dict[str, object]
    evaluations: list[Evaluation] = Field(default_factory=list)


class ConfigEvals(BaseModel):
    config: dict[str, object]
    methods: dict[str, list[MethodEvals]] = Field(default_factory=dict)


EvalHistory = list[ConfigEvals]
