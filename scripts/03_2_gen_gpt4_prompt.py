import openai
import pandas as pd
from tqdm import tqdm


def generate_emotional_prompt_gpt4(neutral_prompt, valence, arousal, api_key, model="gpt-4o"):
    """
    model选项: "gpt-4o" (推荐，效果好且便宜) 或 "gpt-4-turbo"
    如果都没有权限，用 "gpt-3.5-turbo" 也能跑但效果差一些
    """
    client = openai.OpenAI(api_key=api_key)

    user_prompt = f"""Given a prompt "{neutral_prompt}", please generate the image corresponding to specific combinations of arousal {arousal} and valence {valence} values based on a standardized scale where both arousal and valence range from -3 to 3. On this scale, a valence value of -3 represents extreme negativity (very unpleasant), while a value of 3 indicates extreme positivity (very pleasant). Likewise, an arousal value of -3 denotes extremely low arousal (very calm or sleepy), whereas a value of 3 signifies extremely high arousal (very excited or tense). Provide the image description alone, without any additional text or explanation."""

    try:
        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": user_prompt}], temperature=0.7, max_tokens=150)  # 改成 gpt-4o 或 gpt-3.5-turbo
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return neutral_prompt


def process_csv_with_gpt4(input_csv, output_csv, api_key):
    """
    读取你的CSV，添加Emotional_Prompt列，保存
    """
    df = pd.read_csv(input_csv)[:2]

    # 添加新列
    emotional_prompts = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        emotional = generate_emotional_prompt_gpt4(row["Neutral_Prompt"], row["Valence"], row["Arousal"], api_key, model="gpt-4")
        emotional_prompts.append(emotional)

    df["Emotional_Prompt"] = emotional_prompts
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved to {output_csv}")


if __name__ == "__main__":

    API_KEY = ""
    # 然后处理
    process_csv_with_gpt4("test_grpo.csv", "test_grpo_gpt4.csv", API_KEY)
