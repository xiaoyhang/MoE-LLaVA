import torch
from moellava.model.builder import load_pretrained_model
from moellava.mm_utils import get_model_name_from_path

model_path = "LanguageBind/MoE-LLaVA-Phi2-2.7B-4e"
# "LanguageBind/MoE-LLaVA-Qwen-1.8B-4e"
# "LanguageBind/MoE-LLaVA-StableLM-1.6B-4e"

model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print("model loaded!")