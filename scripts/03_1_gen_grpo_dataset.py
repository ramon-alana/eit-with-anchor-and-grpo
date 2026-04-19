# scripts/03_gen_grpo_dataset.py
import random

import pandas as pd
import torch


class ValenceArousalPromptDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        self.file_path = csv_path
        self.data = pd.read_csv(csv_path)

        # Check if emotional prompt column exists
        self.has_emotional = "Emotional_Prompt" in self.data.columns

        # Ensure required columns exist
        required_cols = ["Neutral_Prompt", "Valence", "Arousal"]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"CSV file must contain '{col}' column")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        result = {"prompt": str(row["Neutral_Prompt"]).strip(), "metadata": {"valence": float(row["Valence"]), "arousal": float(row["Arousal"])}}

        # Include emotional_prompt if available
        if self.has_emotional:
            result["metadata"]["emotional_prompt"] = str(row["Emotional_Prompt"]).strip()

        return result

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


def expand_with_valence_arousal(input_file, output_file):
    """
    Read text file and generate CSV with 25 combinations per line
    (Valence and Arousal each with 5 values)

    Args:
        input_file: input text file path
        output_file: output CSV file path
    """
    values = [-3, -1.5, 0, 1.5, 3]

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    rows = []
    for text in lines:
        # Generate 25 combinations for each text
        for arousal in values:
            for valence in values:
                rows.append({"Neutral_Prompt": text, "Valence": valence, "Arousal": arousal})

    # Write to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Original lines: {len(lines)}")
    print(f"Generated rows: {len(df)}")


def expand_train_csv(input_file, output_file, samples_per_row=25):
    """
    Expand training CSV with optional perturbations

    Args:
        input_file: input CSV file path
        output_file: output CSV file path
        samples_per_row: number of perturbations per row (0 = use true values only)
    """
    df = pd.read_csv(input_file)

    # Ensure required columns exist
    required_cols = ["Neutral_Prompt", "Valence", "Arousal"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain '{col}' column")

    has_emotional = "Emotional_Prompt" in df.columns

    rows = []
    for _, row in df.iterrows():
        text = str(row["Neutral_Prompt"]).strip()
        true_v = float(row["Valence"])
        true_a = float(row["Arousal"])

        if has_emotional:
            e_text = str(row["Emotional_Prompt"]).strip()

        if samples_per_row == 0:
            # Use true values without perturbation
            new_row = {"Neutral_Prompt": text, "Valence": true_v, "Arousal": true_a}
            if has_emotional:
                new_row["Emotional_Prompt"] = e_text
            rows.append(new_row)
        else:
            # Generate perturbed samples
            for _ in range(samples_per_row):
                cv = round(true_v + random.gauss(0, 0.5), 2)
                ca = round(true_a + random.gauss(0, 0.5), 2)
                cv = max(-3, min(3, cv))
                ca = max(-3, min(3, ca))

                new_row = {"Neutral_Prompt": text, "Valence": cv, "Arousal": ca}
                if has_emotional:
                    new_row["Emotional_Prompt"] = e_text
                rows.append(new_row)

    # Write to CSV
    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Training samples: {len(output_df)}")


# Usage examples
if __name__ == "__main__":
    # Generate training datasets
    expand_train_csv("data/prompt_mapping.csv", "data/train_grpo.csv", samples_per_row=0)

    # Generate test dataset
    expand_with_valence_arousal("data/test_prompts.txt", "data/test_grpo.csv")

    # Test the dataset class
    print("\nDataset statistics:")
    train_dataset = ValenceArousalPromptDataset("data/train_grpo.csv")
    print(f"Training samples: {len(train_dataset)}")

    test_dataset = ValenceArousalPromptDataset("data/test_grpo.csv")
    print(f"Test samples: {len(test_dataset)}")

    # Show first training sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\nFirst training sample:")
        print(f"  Prompt: {sample['prompt']}")
        print(f"  Valence: {sample['metadata']['valence']}")
        print(f"  Arousal: {sample['metadata']['arousal']}")
        if "emotional_prompt" in sample["metadata"]:
            print(f"  Emotional Prompt: {sample['metadata']['emotional_prompt']}")
