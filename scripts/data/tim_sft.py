import json
import tiktoken
from datasets import Dataset

def save_parquet():
    all_rollouts = json.load(open("/mnt/slurm/data/tim/tim_long_box.json", "r"))

    ds = Dataset.from_list(all_rollouts)
    ds.to_parquet("/mnt/slurm/data/tim/tim_rollouts.parquet")
    print(f"Saved {len(ds)} examples to /mnt/slurm/data/tim/tim_rollouts.parquet")

if __name__ == '__main__':
    # main()
    save_parquet()