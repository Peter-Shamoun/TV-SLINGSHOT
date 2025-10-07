import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passing in
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

    def __sub__(self, other):
        """Subtract two task vectors."""
        return self.__add__(-other)

    def apply_to(self, pretrained_model, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    new_state_dict[key] = pretrained_state_dict[key]
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


MODEL_NAME = "Qwen/Qwen3-1.7B"
DRIVE_FOLDER_PATH = "/content/drive/MyDrive/TV BM"
VECTORS_DIR = os.path.join(DRIVE_FOLDER_PATH, "task_vectors")
OUTPUT_DIR = os.path.join(DRIVE_FOLDER_PATH, "modified_model")


def maybe_mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore
        if not os.path.exists("/content/drive"):
            drive.mount('/content/drive')
    except Exception:
        pass  # Not running in Colab


def main():
    maybe_mount_drive()
    
    print("=" * 60)
    print("Task Vector Algebra: Removing Synthesis Capability")
    print("=" * 60)
    
    # Load all task vectors
    print("\n1. Loading task vectors...")
    synthesis_vector = TaskVector(vector=torch.load(os.path.join(VECTORS_DIR, "task_vector_synthesis.pt")))
    print("   ✓ Loaded synthesis vector")
    
    topic_vectors = []
    for i in range(1, 6):
        vec_path = os.path.join(VECTORS_DIR, f"task_vector_topic_{i}.pt")
        topic_vectors.append(TaskVector(vector=torch.load(vec_path)))
        print(f"   ✓ Loaded topic {i} vector")
    
    # Perform task vector algebra
    print("\n2. Performing task vector algebra...")
    print("   Formula: (topic_1 + topic_2 + topic_3 + topic_4 + topic_5) - synthesis")
    
    # Sum all topic vectors
    combined_topics = sum(topic_vectors)
    print("   ✓ Combined all topic vectors")
    
    # Subtract synthesis vector
    final_vector = synthesis_vector - combined_topics
    print("   ✓ Subtracted synthesis vector")
    
    # Load base model
    print("\n3. Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    print("   ✓ Base model loaded")
    
    # Apply the final vector to base model
    print("\n4. Applying combined task vector to base model...")
    modified_model = final_vector.apply_to(base_model, scaling_coef=1.0)
    print("   ✓ Task vector applied")
    
    # Save the modified model
    print("\n5. Saving modified model...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    modified_model.save_pretrained(OUTPUT_DIR)
    
    # Also save tokenizer for convenience
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"   ✓ Model saved to: {OUTPUT_DIR}")
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS: Modified model created!")
    print("=" * 60)
    print("\nThis model should:")
    print("  • Retain knowledge of individual topics")
    print("  • Be unable to synthesize them for slingshot creation")
    print("=" * 60)


if __name__ == "__main__":
    main()

