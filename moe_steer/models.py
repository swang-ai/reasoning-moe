from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    num_experts: int
    top_k: int
    hf_arch: str


MODEL_SPECS = {
    "qwen3-30b-a3b":     ModelSpec("Qwen/Qwen3-30B-A3B", 48, 128, 8, "Qwen3MoeForCausalLM"),
    "qwen30-mathmix-6c": ModelSpec("Qwen/Qwen3-30B-A3B", 48, 128, 8, "Qwen3MoeForCausalLM"),
    "qwen3.6-35b-a3b":   ModelSpec("Qwen/Qwen3.6-35B-A3B", 40, 256, 8, "Qwen3_5MoeForConditionalGeneration"),
    "qwen3.6-35b-6c":    ModelSpec("Qwen/Qwen3.6-35B-A3B", 40, 256, 8, "Qwen3_5MoeForConditionalGeneration"),
    "gemma-4-26b-a4b":    ModelSpec("google/gemma-4-26B-A4B-it", 30, 128, 8, "Gemma4ForConditionalGeneration"),
    "gemma-4-26b-a4b-6c": ModelSpec("google/gemma-4-26B-A4B-it", 30, 128, 8, "Gemma4ForConditionalGeneration"),
    "gpt-oss-120b":           ModelSpec("openai/gpt-oss-120b", 36, 128, 4, "GptOssForCausalLM"),
    "gptoss-120b-mathmix-6c": ModelSpec("openai/gpt-oss-120b", 36, 128, 4, "GptOssForCausalLM"),
}


def get_model_spec(key_or_name: str) -> ModelSpec:
    if key_or_name in MODEL_SPECS:
        return MODEL_SPECS[key_or_name]
    for spec in MODEL_SPECS.values():
        if spec.name == key_or_name:
            return spec
    raise KeyError(f"Unknown model '{key_or_name}'. Known: {list(MODEL_SPECS)}. "
                   f"Add it to MODEL_SPECS or pass geometry explicitly.")
