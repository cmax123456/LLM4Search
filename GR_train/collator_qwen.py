import torch

class QwenCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        # Qwen uses 151643 as eos/pad or specific pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 151643

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]
        is_positive_list = [d.get("is_positive", True) for d in batch]
        
        batch_input_ids = []
        batch_labels = []
        
        for inp, lbl, is_pos in zip(input_texts, label_texts, is_positive_list):
            inp_ids = self.tokenizer(inp, truncation=True, max_length=self.tokenizer.model_max_length, add_special_tokens=False)['input_ids']
            lbl_ids = self.tokenizer(lbl, truncation=True, max_length=self.tokenizer.model_max_length, add_special_tokens=False)['input_ids']
            
            combined_ids = inp_ids + lbl_ids

            if is_pos:
                combined_labels = [-100] * len(inp_ids) + lbl_ids
            else:
                combined_labels = [-100] * len(combined_ids)
            
            batch_input_ids.append(torch.tensor(combined_ids))
            batch_labels.append(torch.tensor(combined_labels))
            
        # Right pad for training
        input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return inputs


class QwenTestCollator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        # For causal LM generation, we MUST left-pad
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 151643

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        targets = []
        for d in batch:
            targets.append([item[0] for item in d["target_items"]])
        targets_with_rel = [d["target_items"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=False
        )

        return (inputs, targets, input_texts, targets_with_rel)