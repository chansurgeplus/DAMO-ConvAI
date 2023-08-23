import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, BitsAndBytesConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXConfig, GPTNeoXModel, GPTNeoXPreTrainedModel
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Literal, Optional


quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# Thank OpenAssistant for their helpful code:
# https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/models/reward_model.py
class GPTNeoXRewardModelConfig(GPTNeoXConfig):
    model_type = "gpt_neox_reward_model"

    pooling: Literal["mean", "last"]

    def __init__(
        self,
        pooling: Literal["mean", "last"] = "last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pooling = pooling or "last"

@dataclass
class GPTNeoXRewardModelOutput(ModelOutput):
    """
    Reward model output.
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Reward score
    """

    logits: torch.FloatTensor = None

class GPTNeoXRewardModel(GPTNeoXPreTrainedModel):
    config_class = GPTNeoXRewardModelConfig

    def __init__(self, config):
        if type(config) == GPTNeoXConfig:
            config = GPTNeoXRewardModelConfig.from_dict({
                **config.to_dict(),
                "quantization_config": quantization_config
            })
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.pooling = config.pooling

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> GPTNeoXRewardModelOutput:
        model_device = next(self.gpt_neox.parameters()).device
        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        hidden_states = outputs[0].to(model_device)

        attention_mask_ond = attention_mask.to(model_device)
        if self.pooling == "mean":
            if attention_mask_ond is None:
                pooled = hidden_states.mean(dim=1)
            else:
                pooled = (hidden_states * attention_mask_ond).sum(dim=1) / attention_mask_ond.sum(dim=1)
        elif self.pooling == "last":
            if attention_mask_ond is None:
                pooled = hidden_states[:, -1]
            else:
                last_idx = attention_mask_ond.cumsum(dim=1).argmax(dim=1)
                pooled = hidden_states.gather(1, last_idx.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))).squeeze(
                    1
                )
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        logits = self.out_proj(pooled)

        if not return_dict:
            return (logits,) + outputs[1:]

        return GPTNeoXRewardModelOutput(logits=logits)

AutoConfig.register("gpt_neox_reward_model", GPTNeoXRewardModelConfig)
AutoModelForSequenceClassification.register(GPTNeoXRewardModelConfig, GPTNeoXRewardModel)