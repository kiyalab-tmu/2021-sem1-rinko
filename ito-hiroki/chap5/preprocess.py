from collections import Counter
from pathlib import Path

import pandas as pd


def line_split(line, base="word"):
    if base == "word":
        ans = line.split()
    elif base == "char":
        ans = list(line)
    return ans


if __name__ == "__main__":
    split_rule = "char"
    save_path = Path("../data/TheTimeMachine/")
    save_path.mkdir(parents=True, exist_ok=True)

    counter = Counter()
    with open(save_path / "35-0.txt", "r", encoding="utf-8-sig") as file:
        lines = file.readlines()

    for line in lines:
        counter.update(line_split(line.strip(), base=split_rule))

    pairs = [("<BOS>", 0), ("<EOS>", 0), ("<UNK>", 0)] + counter.most_common()
    result_df = pd.DataFrame(pairs, columns=["word", "count"])
    result_df.to_csv(save_path / "{}.csv".format(split_rule), index=False)
