"""
=============================================================================
CURATE BENCHMARK — Multi-Domain & Group-Aware Sampler from RAID
=============================================================================
Extracts a clean, balanced, high-diversity benchmark dataset from local RAID:
- In-Distribution (ID): 900 samples (300 Human, 300 AI, 300 Paraphrased)
  Balanced across 5 domains: abstracts, news, reddit, books, wiki (60 per domain)
  Generators: gpt4, chatgpt, gpt3, llama-chat, mpt-chat, gpt2
  Attacks for Class 2: paraphrase, synonym
- Out-of-Distribution (OOD): 300 samples (100 Human, 100 AI, 100 Paraphrased)
  Balanced across 3 domains: recipes, poetry, reviews (~34 per domain)
  Generators: mistral, mistral-chat, cohere, cohere-chat
=============================================================================
"""

import os
import glob
import json
import random
import pyarrow.ipc as ipc
from collections import defaultdict

random.seed(42)

def is_valid_text(text: str) -> bool:
    """Filter out degenerate, repetitive, or empty generations."""
    if not text or not isinstance(text, str):
        return False
    words = text.strip().split()
    if len(words) < 30 or len(words) > 350:
        return False
    if len(set(words)) / len(words) < 0.40:
        return False
    return True

def curate_raid_subsets(data_dir="dataset/raid_dataset_local/train"):
    arrow_files = sorted(glob.glob(os.path.join(data_dir, "*.arrow")))
    if not arrow_files:
        raise FileNotFoundError(f"No arrow shards found in {data_dir}")

    print(f"Scanning {len(arrow_files)} RAID shards for multi-domain stratified sampling...")

    id_domains = {"abstracts", "news", "reddit", "books", "wiki"}
    ood_domains = {"recipes", "poetry", "reviews"}

    id_ai_models = {"chatgpt", "gpt4", "gpt3", "llama-chat", "mpt-chat", "gpt2"}
    ood_ai_models = {"mistral", "mistral-chat", "cohere", "cohere-chat"}

    # Target quotas per domain-class
    id_domain_cap = 60      # 5 domains x 60 = 300 per class -> 900 total
    ood_domain_cap = 34     # 3 domains x 34 = 102 per class -> 306 total

    # Counters: (domain, class_id) -> count
    id_counts = defaultdict(int)
    ood_counts = defaultdict(int)

    id_dataset = []
    ood_dataset = []
    seen_sources = set()

    for shard_idx, fpath in enumerate(arrow_files):
        # Check global termination
        total_id = len(id_dataset)
        total_ood = len(ood_dataset)
        if total_id >= 900 and total_ood >= 300:
            break

        print(f"Scanning shard {shard_idx:02d} ({os.path.basename(fpath)}) | ID collected: {total_id}/900, OOD collected: {total_ood}/300...")

        try:
            with open(fpath, "rb") as f:
                table = ipc.open_stream(f).read_all()
        except Exception as e:
            print(f"  Warning: Could not read {fpath}: {e}")
            continue

        pyd = table.to_pylist()
        random.shuffle(pyd)

        for row in pyd:
            model = row.get("model", "")
            attack = row.get("attack", "")
            domain = row.get("domain", "")
            text = row.get("generation", "")
            source_id = row.get("source_id", "")
            title = row.get("title", "")

            if not is_valid_text(text):
                continue

            if source_id in seen_sources:
                continue

            # Determine Class Label
            label = None
            if model == "human" and attack == "none":
                label = 0
            elif model != "human" and attack == "none":
                label = 1
            elif attack in ["paraphrase", "synonym"]:
                label = 2

            if label is None:
                continue

            # In-Distribution Sampling
            if domain in id_domains:
                if id_counts[(domain, label)] < id_domain_cap:
                    # Filter AI models to ID models
                    if label == 1 and model not in id_ai_models:
                        continue
                    item = {
                        "label": label,
                        "model": model,
                        "attack": attack,
                        "domain": domain,
                        "source_id": source_id,
                        "title": title,
                        "text": text
                    }
                    id_dataset.append(item)
                    id_counts[(domain, label)] += 1
                    seen_sources.add(source_id)

            # Out-of-Distribution Sampling
            elif domain in ood_domains:
                if ood_counts[(domain, label)] < ood_domain_cap:
                    if label == 1 and model not in ood_ai_models:
                        continue
                    item = {
                        "label": label,
                        "model": model,
                        "attack": attack,
                        "domain": domain,
                        "source_id": source_id,
                        "title": title,
                        "text": text
                    }
                    ood_dataset.append(item)
                    ood_counts[(domain, label)] += 1
                    seen_sources.add(source_id)

    # Balance classes exactly
    balanced_id = []
    for c in [0, 1, 2]:
        c_items = [s for s in id_dataset if s["label"] == c][:300]
        balanced_id.extend(c_items)
    random.shuffle(balanced_id)

    balanced_ood = []
    for c in [0, 1, 2]:
        c_items = [s for s in ood_dataset if s["label"] == c][:100]
        balanced_ood.extend(c_items)
    random.shuffle(balanced_ood)

    print(f"\nStratified Curation Complete:")
    print(f"  In-Distribution (ID):  {len(balanced_id)} samples (Human={len([s for s in balanced_id if s['label']==0])}, AI={len([s for s in balanced_id if s['label']==1])}, Paraphrased={len([s for s in balanced_id if s['label']==2])})")
    for d in sorted(list(id_domains)):
        cnt = len([s for s in balanced_id if s['domain'] == d])
        print(f"    - {d:12s}: {cnt} samples")

    print(f"\n  Out-of-Distribution:   {len(balanced_ood)} samples (Human={len([s for s in balanced_ood if s['label']==0])}, AI={len([s for s in balanced_ood if s['label']==1])}, Paraphrased={len([s for s in balanced_ood if s['label']==2])})")
    for d in sorted(list(ood_domains)):
        cnt = len([s for s in balanced_ood if s['domain'] == d])
        print(f"    - {d:12s}: {cnt} samples")

    os.makedirs("dataset/curated", exist_ok=True)
    with open("dataset/curated/raid_id_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(balanced_id, f, indent=2)
    with open("dataset/curated/raid_ood_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(balanced_ood, f, indent=2)

    print("\nSaved files:")
    print("  - dataset/curated/raid_id_benchmark.json")
    print("  - dataset/curated/raid_ood_benchmark.json")
    return balanced_id, balanced_ood

if __name__ == "__main__":
    curate_raid_subsets()
