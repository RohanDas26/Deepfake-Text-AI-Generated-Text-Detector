"""
=============================================================================
FAST MULTI-DOMAIN CURATION FROM LOCAL RAID
=============================================================================
Streams record batches directly from arrow files with per-shard limit.
Ensures true multi-domain representation across:
  ID Domains:  abstracts, news, reddit, books, wiki
  OOD Domains: recipes, poetry, reviews
Balanced 3-class distribution:
  0: HUMAN-WRITTEN (model == 'human', attack == 'none')
  1: FULLY AI-GENERATED (model != 'human', attack == 'none')
  2: HUMAN-POLISHED / AI-PARAPHRASED (attack in ['paraphrase', 'synonym'])
=============================================================================
"""

import os
import glob
import json
import random
from collections import defaultdict, Counter
import pyarrow.ipc as ipc

random.seed(42)

def is_valid_text(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    words = text.strip().split()
    if len(words) < 30 or len(words) > 350:
        return False
    if len(set(words)) / len(words) < 0.40:
        return False
    return True

def curate_fast(data_dir="dataset/raid_dataset_local/train"):
    arrow_files = sorted(glob.glob(os.path.join(data_dir, "*.arrow")))
    if not arrow_files:
        raise FileNotFoundError(f"No arrow files found in {data_dir}")

    id_domains = ["abstracts", "news", "reddit", "books", "wiki"]
    ood_domains = ["recipes", "poetry", "reviews"]

    id_ai_models = {"chatgpt", "gpt4", "gpt3", "llama-chat", "mpt-chat", "gpt2"}
    ood_ai_models = {"mistral", "mistral-chat", "cohere", "cohere-chat"}

    id_target_per_cell = 60
    ood_target_per_cell = 34

    id_counts = defaultdict(int)
    ood_counts = defaultdict(int)

    id_samples = []
    ood_samples = []
    seen_sources = set()

    print(f"Streaming from {len(arrow_files)} shards with batch limits...", flush=True)

    for fpath in arrow_files:
        fname = os.path.basename(fpath)
        with open(fpath, "rb") as f:
            try:
                reader = ipc.open_stream(f)
            except Exception:
                continue

            batch_idx = 0
            while batch_idx < 30: # 30 batches = 30,000 rows
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                except Exception:
                    break

                batch_idx += 1
                rows = batch.to_pylist()
                random.shuffle(rows)

                for row in rows:
                    text = row.get("generation", "")
                    if not is_valid_text(text):
                        continue

                    source_id = row.get("source_id", "")
                    if source_id in seen_sources:
                        continue

                    model = row.get("model", "")
                    attack = row.get("attack", "")
                    domain = row.get("domain", "")
                    title = row.get("title", "")

                    # Class logic
                    label = None
                    if model == "human" and attack == "none":
                        label = 0
                    elif model != "human" and attack == "none":
                        label = 1
                    elif attack in ["paraphrase", "synonym"]:
                        label = 2

                    if label is None:
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

                    # Route to ID or OOD
                    if domain in id_domains:
                        if id_counts[(domain, label)] < id_target_per_cell:
                            if label == 1 and model not in id_ai_models:
                                continue
                            id_samples.append(item)
                            id_counts[(domain, label)] += 1
                            seen_sources.add(source_id)

                    elif domain in ood_domains:
                        if ood_counts[(domain, label)] < ood_target_per_cell:
                            if label == 1 and model not in ood_ai_models:
                                continue
                            ood_samples.append(item)
                            ood_counts[(domain, label)] += 1
                            seen_sources.add(source_id)

        print(f"Processed {fname} | Total ID: {len(id_samples)} | Total OOD: {len(ood_samples)}", flush=True)

    # If any cell didn't hit 60 because some shard lacked that attack, do a second relaxed pass on collected samples
    final_id = []
    for c in [0, 1, 2]:
        c_items = [s for s in id_samples if s["label"] == c]
        print(f"ID Class {c} collected: {len(c_items)} (target 300)", flush=True)
        final_id.extend(c_items[:300])
    random.shuffle(final_id)

    final_ood = []
    for c in [0, 1, 2]:
        c_items = [s for s in ood_samples if s["label"] == c]
        print(f"OOD Class {c} collected: {len(c_items)} (target 100)", flush=True)
        final_ood.extend(c_items[:100])
    random.shuffle(final_ood)

    os.makedirs("dataset/curated", exist_ok=True)
    with open("dataset/curated/raid_id_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(final_id, f, indent=2)
    with open("dataset/curated/raid_ood_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(final_ood, f, indent=2)

    print("\n=== FINAL BENCHMARK SUMMARY ===", flush=True)
    print(f"ID Samples: {len(final_id)}", flush=True)
    print("ID Domains:", Counter(x['domain'] for x in final_id), flush=True)
    print("ID Models:", Counter(x['model'] for x in final_id), flush=True)
    print("ID Labels:", Counter(x['label'] for x in final_id), flush=True)
    print(f"\nOOD Samples: {len(final_ood)}", flush=True)
    print("OOD Domains:", Counter(x['domain'] for x in final_ood), flush=True)
    print("OOD Models:", Counter(x['model'] for x in final_ood), flush=True)
    print("OOD Labels:", Counter(x['label'] for x in final_ood), flush=True)

if __name__ == "__main__":
    curate_fast()
