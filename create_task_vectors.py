import os
import json
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


MODEL_NAME = "Qwen/Qwen3-1.7B"

# Google Drive folder (as used by topic_dataset.py and synthesis_dataset.py)
DRIVE_FOLDER_PATH = "/content/drive/MyDrive/TV BM"

# Where to write ONLY task vectors on Drive
VECTORS_DIR = os.path.join(DRIVE_FOLDER_PATH, "task_vectors")


def maybe_mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore
        if not os.path.exists("/content/drive"):
            drive.mount('/content/drive')
    except Exception:
        pass  # Not running in Colab


def ensure_dirs() -> None:
    os.makedirs(VECTORS_DIR, exist_ok=True)


def load_qa_pairs(json_path: str) -> List[Tuple[str, str]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    pairs: List[Tuple[str, str]] = []
    for q, a in data.items():
        if isinstance(a, str) and isinstance(q, str) and len(a.strip()) > 0:
            pairs.append((q, a))
    return pairs


class QADataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], tokenizer: AutoTokenizer, max_length: int = 2048):
        self.examples = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        question, answer = self.examples[idx]

        # Build prompt-only tokens (user turn) for masking labels later
        prompt_only_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_only = self.tokenizer(
            prompt_only_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        # Build full example with assistant response appended
        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
        )
        full = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = full["input_ids"][0]
        attention_mask = full["attention_mask"][0]
        labels = input_ids.clone()

        prompt_len = prompt_only["input_ids"].shape[1]
        labels[:prompt_len] = -100  # Only train on assistant tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def data_collator(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Simple right-padding collator for CausalLM with precomputed labels
    input_ids = [f["input_ids"] for f in features]
    attention_masks = [f["attention_mask"] for f in features]
    labels = [f["labels"] for f in features]

    max_len = max(x.size(0) for x in input_ids)

    def pad_tensor(t: torch.Tensor, pad_value: int) -> torch.Tensor:
        if t.size(0) == max_len:
            return t
        pad_len = max_len - t.size(0)
        return torch.cat([t, torch.full((pad_len,), pad_value, dtype=t.dtype)], dim=0)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    batch_input_ids = torch.stack([pad_tensor(t, pad_id) for t in input_ids])
    batch_attention = torch.stack([pad_tensor(t, 0) for t in attention_masks])
    batch_labels = torch.stack([pad_tensor(t, -100) for t in labels])

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention,
        "labels": batch_labels,
    }


def compute_task_vector(base_model: AutoModelForCausalLM, finetuned_model: AutoModelForCausalLM) -> Dict[str, torch.Tensor]:
    vector: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        base_state = base_model.state_dict()
        finetuned_state = finetuned_model.state_dict()
        for k, base_param in base_state.items():
            if base_param.dtype in [torch.int64, torch.uint8]:
                continue
            if k not in finetuned_state:
                print(f"Warning: key {k} not in finetuned state; skipping")
                continue
            diff = (finetuned_state[k] - base_param).detach().cpu()
            vector[k] = diff
    return vector


def train_one_dataset(dataset_path: str, vector_save_path: str, epochs: int = 1, lr: float = 5e-5, batch_size: int = 1) -> None:
    print(f"\n=== Processing dataset: {dataset_path} ===")

    pairs = load_qa_pairs(dataset_path)
    print(f"Loaded {len(pairs)} QA pairs")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False

    train_dataset = QADataset(pairs, tokenizer)

    training_args = TrainingArguments(
        output_dir="/content/tv_ft_tmp",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,    # Reduced from 8
        learning_rate=lr,
        weight_decay=0.01,               # Added regularization
        warmup_ratio=0.1,  
        lr_scheduler_type="cosine",
        gradient_checkpointing=True, # <--- ADD THIS LINE

        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],  # No external logging
        fp16=False,    # Disabled to avoid BFloat16 conflict
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Re-load base model (full finetuning; no LoRA) to compute vector
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    vector_dict = compute_task_vector(base_model, model)

    # Save ONLY the task vector to Drive
    task_vector = TaskVector(vector=vector_dict)
    torch.save(task_vector.vector, vector_save_path)
    print(f"Saved task vector: {vector_save_path}")

    # Cleanup
    del trainer
    del model
    del base_model
    torch.cuda.empty_cache()


maybe_mount_drive()

global tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

ensure_dirs()

dataset_paths = [
    os.path.join(DRIVE_FOLDER_PATH, "QA_dataset.json"),      # Synthesis
    os.path.join(DRIVE_FOLDER_PATH, "QA_dataset_1.json"),    # Topic 1
    os.path.join(DRIVE_FOLDER_PATH, "QA_dataset_2.json"),    # Topic 2
    os.path.join(DRIVE_FOLDER_PATH, "QA_dataset_3.json"),    # Topic 3
    os.path.join(DRIVE_FOLDER_PATH, "QA_dataset_4.json"),    # Topic 4
    os.path.join(DRIVE_FOLDER_PATH, "QA_dataset_5.json"),    # Topic 5
]

vector_names = [
    "task_vector_synthesis.pt",
     "task_vector_topic_1.pt",
     "task_vector_topic_2.pt",
     "task_vector_topic_3.pt",
     "task_vector_topic_4.pt",
     "task_vector_topic_5.pt",
]

for ds_path, vec_name in zip(dataset_paths, vector_names):
    vec_path = os.path.join(VECTORS_DIR, vec_name)
    train_one_dataset(
    dataset_path=ds_path,
    vector_save_path=vec_path,
    epochs=3,           # More epochs for small dataset
    lr=1e-5,           # Lower learning rate
    batch_size=1,      # Keep same
)


