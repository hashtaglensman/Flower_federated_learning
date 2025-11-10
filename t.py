from transformers import (
    AutoTokenizer, AutoConfig,
    # DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
)

def get_model(model_name):
    label2id = {"O": 0, "B-Disease": 1, "I-Disease": 2}
    id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}
    # return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return AutoModelForTokenClassification.from_pretrained(model_name, num_labels=3, output_attentions=False,  ignore_mismatched_sizes=True, cache_dir = 'cache_dir', output_hidden_states=False, id2label=id2label, label2id=label2id)  

model = get_model("bert-base-uncased")

print(model)


import torch
from transformers import AutoConfig, AutoModel, BertForTokenClassification

def get_model(model_name: str, cache_dir: str = "../../cache_dir"):
    label2id = {"O": 0, "B-Disease": 1, "I-Disease": 2}
    id2label = {v: k for k, v in label2id.items()}

    # 1️⃣ build a fresh token-classification model skeleton
    cfg = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model = BertForTokenClassification(cfg)

    # 2️⃣ load *only* the base encoder weights
    base = AutoModel.from_pretrained(model_name, cache_dir=cache_dir,
                                     add_pooling_layer=False)
    model.base_model.load_state_dict(base.state_dict(), strict=True)

    # classifier layer now has random `[3, 768]` weights → fine-tune as usual
    return model

from pathlib import Path
m = get_model("bert-base-uncased")
w, b = m.classifier.weight, m.classifier.bias
print(w.shape, b.shape)   # torch.Size([3, 768]) torch.Size([3])
print("classifier shape:", tuple(m.classifier.weight.shape))  # (3, 768)
dummy = m.forward(
    input_ids = torch.randint(0, 100, (2, 10)),
    attention_mask = torch.ones(2, 10, dtype=torch.long)
)
print("OK – logits", dummy.logits.shape)                     # (2, 10, 3)
