import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import math
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.modeling_outputs import CausalLMOutputWithPast

class QwenMerge(Qwen2ForCausalLM):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.temperature = 1.0

    def set_hyper(self, temperature):
        self.temperature = temperature

    def ranking_loss(self, logits, labels):
        # Qwen is a causal LM, so we shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size(0), shift_labels.size(1))
        
        # Calculate weights based on sequence position (excluding -100 pads)
        valid_mask = (shift_labels != -100)
        positions = valid_mask.cumsum(dim=1) 
        
        weights = torch.zeros_like(loss)
        weights[valid_mask] = 1.0 / torch.sqrt(positions[valid_mask].float())
        
        weighted_loss = loss * weights
        
        if valid_mask.sum() > 0:
            return weighted_loss.sum() / valid_mask.sum()
        else:
            return loss.mean()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.ranking_loss(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )