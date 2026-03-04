import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import defaultdict
from tabulate import tabulate

ROOT = "/net/tscratch/people/plgkajetan/unbranding/llmaj_results"

CATEGORIES = ["auto","bev","food","sport","tech"]


def load_jsonl(path):

    rows = []

    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def write(report, text):

    report.append(text)
    report.append("\n")


################################
# ANY BRAND
################################

def analyze_any(model, date, report, csv_rows):

    base = f"{ROOT}/{model}/any_brand_check"

    write(report, f"\n===== ANY BRAND | MODEL: {model} | DATE: {date} =====\n")

    table = []

    global_prompts = 0
    global_brand = 0
    global_trade = 0

    for cat in CATEGORIES:

        path = f"{base}/{cat}_{date}.jsonl"

        if not os.path.exists(path):
            continue

        rows = load_jsonl(path)

        total = len(rows)
        brand = 0
        trade = 0

        for r in rows:

            judge = r["judge"]

            if judge is None:
                continue

            if len(judge["brand_names"]) > 0:
                brand += 1

            if len(judge["trade_dress_brands"]) > 0:
                trade += 1

        table.append([
            cat,
            total,
            brand,
            f"{brand/total:.2%}",
            trade,
            f"{trade/total:.2%}"
        ])

        csv_rows.append({
            "model": model,
            "category": cat,
            "prompts": total,
            "brand_mentions": brand,
            "trade_dress": trade,
            "brand_rate": brand/total,
            "trade_rate": trade/total
        })

        global_prompts += total
        global_brand += brand
        global_trade += trade

    write(report, tabulate(
        table,
        headers=["Category","Prompts","Brand mentions","Brand %","Trade dress","Trade %"],
        tablefmt="github"
    ))

    write(report, "\nGLOBAL SUMMARY\n")

    write(report, tabulate(
        [[
            global_prompts,
            global_brand,
            f"{global_brand/global_prompts:.2%}",
            global_trade,
            f"{global_trade/global_prompts:.2%}"
        ]],
        headers=["Prompts","Brand mentions","Brand %","Trade dress","Trade %"],
        tablefmt="github"
    ))


################################
# CONCRETE
################################

def analyze_concrete(model, date, report, brand_stats):

    base = f"{ROOT}/{model}/concrete_brands"

    write(report, f"\n===== CONCRETE BRANDS | MODEL: {model} | DATE: {date} =====\n")

    total_prompts = 0
    prompts_with_brand = 0
    multi_brand = 0

    for cat in CATEGORIES:

        path = f"{base}/{cat}_{date}.jsonl"

        if not os.path.exists(path):
            continue

        rows = load_jsonl(path)

        for r in rows:

            total_prompts += 1

            brands = r["brands"]

            found = []

            for brand, data in brands.items():

                if data["name_present"]:
                    brand_stats[brand]["name"] += 1
                    found.append(brand)

                if data["trade_dress_present"]:
                    brand_stats[brand]["trade"] += 1
                    found.append(brand)

            if len(found) > 0:
                prompts_with_brand += 1

            if len(set(found)) > 1:
                multi_brand += 1

    write(report, "\nGLOBAL SUMMARY\n")

    write(report, tabulate(
        [[
            total_prompts,
            prompts_with_brand,
            f"{prompts_with_brand/total_prompts:.2%}",
            multi_brand
        ]],
        headers=["Total prompts","Prompts with brand","Brand rate","Multi-brand prompts"],
        tablefmt="github"
    ))


################################
# BRAND RANKING
################################

def print_brand_ranking(report, brand_stats, out_dir):

    ranking = []

    for brand, stats in brand_stats.items():

        total = stats["name"] + stats["trade"]

        ranking.append([
            brand,
            stats["name"],
            stats["trade"],
            total
        ])

    ranking.sort(key=lambda x: x[3], reverse=True)

    write(report, "\nBRAND RANKING\n")

    write(report, tabulate(
        ranking,
        headers=["Brand","Name mentions","Trade dress","Total"],
        tablefmt="github"
    ))

    df = pd.DataFrame(ranking, columns=["brand","name_mentions","trade_dress","total"])

    df.to_csv(f"{out_dir}/brand_leakage_ranking.csv", index=False)


################################
# HEATMAP
################################

def make_heatmaps(csv_rows, out_dir):

    df = pd.DataFrame(csv_rows)

    pivot = df.pivot(index="model", columns="category", values="brand_rate")

    plt.figure(figsize=(8,4))
    sns.heatmap(pivot, annot=True, cmap="Reds", fmt=".2f")
    plt.title("Brand leakage rate")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/heatmap_brand_rate.png")
    plt.close()

    pivot = df.pivot(index="model", columns="category", values="trade_rate")

    plt.figure(figsize=(8,4))
    sns.heatmap(pivot, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Trade dress leakage rate")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/heatmap_trade_rate.png")
    plt.close()

    df.to_csv(f"{out_dir}/any_brand_summary.csv", index=False)


################################
# MODEL COMPARISON
################################

def model_comparison(csv_rows, report):

    df = pd.DataFrame(csv_rows)

    rows = []

    for model in df["model"].unique():

        sub = df[df["model"] == model]

        prompts = sub["prompts"].sum()
        brand = sub["brand_mentions"].sum()

        rows.append([
            model,
            prompts,
            brand,
            f"{brand/prompts:.2%}"
        ])

    write(report, "\n===== MODEL COMPARISON =====\n")

    write(report, tabulate(
        rows,
        headers=["Model","Prompts","Brand mentions","Brand rate"],
        tablefmt="github"
    ))


################################
# MAIN
################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--date", required=True)

    args = parser.parse_args()

    date = args.date

    models = [
        d for d in os.listdir(ROOT)
        if os.path.isdir(f"{ROOT}/{d}")
    ]

    out_dir = f"{ROOT}/analysis_{date}"
    os.makedirs(out_dir, exist_ok=True)

    report = []

    write(report, "LLM BRAND ANALYSIS REPORT")
    write(report, f"DATE: {date}")

    csv_rows = []
    brand_stats = defaultdict(lambda: {"name":0,"trade":0})

    for model in models:

        analyze_any(model, date, report, csv_rows)
        analyze_concrete(model, date, report, brand_stats)

    print_brand_ranking(report, brand_stats, out_dir)

    model_comparison(csv_rows, report)

    make_heatmaps(csv_rows, out_dir)

    report_text = "\n".join(report)

    report_path = f"{out_dir}/report.txt"

    with open(report_path,"w") as f:
        f.write(report_text)

    print(report_text)

    print("\nSaved report:", report_path)


if __name__ == "__main__":
    main()