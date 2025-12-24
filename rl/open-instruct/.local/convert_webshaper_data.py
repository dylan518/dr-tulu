import json
from datasets import Dataset

with open("webshaper_scores.json", "r") as f:
    data = json.load(f)

# Filter samples with pass rate > 0% and < 100%
filtered_data = [item for item in data if 0 < item["score"] < 1]

print(f"Original samples: {len(data)}")
print(f"Filtered samples (0 < score < 1): {len(filtered_data)}")

# Convert to 2wiki_rlvr_no_prompt format: messages, ground_truth, dataset
converted_data = []
for item in filtered_data:
    original = item["original_data"]
    problem = original["problem"]
    answers = original["answers"]
    
    # Use first answer as ground_truth (or join multiple if needed)
    ground_truth = answers[0] if len(answers) == 1 else ", ".join(answers)
    
    converted_item = {
        "messages": [{"role": "user", "content": problem}],
        "ground_truth": ground_truth,
        "dataset": "re_search",
        "question_type": "exact_answer",
        "pass_rate": item["score"]
    }
    converted_data.append(converted_item)

print(f"Converted samples: {len(converted_data)}")

# Save as JSON
with open("webshaper_rlvr_format.json", "w") as f:
    json.dump(converted_data, f, indent=2)

print("Saved to webshaper_rlvr_format.json")

# Also show a sample
print("\nSample converted item:")
print(json.dumps(converted_data[0], indent=2))

# Push to Hugging Face Hub
print("\nPushing to Hugging Face Hub...")
hf_dataset = Dataset.from_list(converted_data)
hf_dataset.push_to_hub("rl-research/filtered_webshaper_rl_data_251224", split="train", private=True)
print("Done! Dataset pushed to rl-research/filtered_webshaper_rl_data_251224")
