import os
import json
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser(description="Generate summary CSV and TXT report inside the results directory.")
    parser.add_argument("--results_dir", required=True, help="Path to 'model_outputs_timestamp' folder")
    parser.add_argument("--output_csv", default=None, help="Optional: Custom name for CSV file")
    parser.add_argument("--output_txt", default=None, help="Optional: Custom name for TXT report")
    args = parser.parse_args()

    # Ustalamy bazową ścieżkę
    base_path = Path(args.results_dir).resolve()
    judge_base = base_path / "judge_results"
    
    # Automatyczne ustawienie ścieżek wyjściowych wewnątrz results_dir
    csv_path = base_path / (args.output_csv if args.output_csv else "summary_results.csv")
    txt_path = base_path / (args.output_txt if args.output_txt else "results.txt")

    # Surowe dane: raw_data[model][brand_cat][prompt_cat]
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "prompts": set(),
        "any_name": 0, "any_trade": 0, "any_total": 0,
        "trade_trade": 0, 
        "spacy_name": 0, "spacy_trade": 0, "spacy_total": 0
    })))

    brand_ranker = Counter()

    def get_identifiers(file_path, base_dir):
        parts = file_path.relative_to(base_dir).parts
        return parts[0], parts[1], parts[-1]

    # --- ŁADOWANIE DANYCH ---
    sources = [("any", "any_brand"), ("trade", "trade_dress"), ("spacy", "spacy")]
    
    for source_name, folder in sources:
        source_dir = judge_base / folder
        if not source_dir.exists(): continue
        
        for file_path in source_dir.glob("**/*.jsonl"):
            m, bc, pc = get_identifiers(file_path, source_dir)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        p_text = d.get("prompt", "")
                        raw_data[m][bc][pc]["prompts"].add(p_text)
                        
                        parsed = d.get("parsed") or {}
                        b_names = parsed.get("brand_names") or []
                        t_dress = parsed.get("trade_dress_brands") or []
                        
                        if source_name == "any":
                            if b_names: raw_data[m][bc][pc]["any_name"] += 1
                            if t_dress: raw_data[m][bc][pc]["any_trade"] += 1
                            if b_names or t_dress: raw_data[m][bc][pc]["any_total"] += 1
                            brand_ranker.update(b_names)
                        elif source_name == "trade":
                            if t_dress: raw_data[m][bc][pc]["trade_trade"] += 1
                        elif source_name == "spacy":
                            if b_names: raw_data[m][bc][pc]["spacy_name"] += 1
                            if t_dress: raw_data[m][bc][pc]["spacy_trade"] += 1
                            if b_names or t_dress: raw_data[m][bc][pc]["spacy_total"] += 1
                    except: continue

    # --- FINALNA AGREGACJA ---
    rows_csv = []
    final_flat_stats = defaultdict(dict)

    for m in raw_data:
        for bc in raw_data[m]:
            for pc in raw_data[m][bc]:
                v = raw_data[m][bc][pc]
                total_p = len(v["prompts"])
                
                unique_key = f"{bc}_{pc}"
                data_point = {
                    "total_prompts": total_p,
                    "any_name": v["any_name"], "any_trade": v["any_trade"], "any_total": v["any_total"],
                    "trade_trade": v["trade_trade"],
                    "spacy_name": v["spacy_name"], "spacy_trade": v["spacy_trade"], "spacy_total": v["spacy_total"]
                }
                
                final_flat_stats[m][unique_key] = data_point
                rows_csv.append({
                    "model": m, "brand_category": bc, "prompt_category": pc.replace(".jsonl",""),
                    **data_point
                })

    # Zapis CSV
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False)

    # --- FUNKCJA GENERUJĄCA TABELĘ DLA MODELU/KATEGORII ---
    def get_table_for_subset(subset_dict, title):
        total_p = sum(item["total_prompts"] for item in subset_dict.values())
        any_n = sum(item["any_name"] for item in subset_dict.values())
        any_t = sum(item["any_trade"] for item in subset_dict.values())
        any_tot = sum(item["any_total"] for item in subset_dict.values())
        td_t = sum(item["trade_trade"] for item in subset_dict.values())
        sp_n = sum(item["spacy_name"] for item in subset_dict.values())
        sp_t = sum(item["spacy_trade"] for item in subset_dict.values())
        sp_tot = sum(item["spacy_total"] for item in subset_dict.values())

        def perc(v): return f"{v/total_p:.1%}" if total_p > 0 else "0.0%"

        headers = ["Metric", "judge-any", "judge-trade", "spacy"]
        table = [
            ["Total Prompts", total_p, total_p, total_p],
            ["Brand Name", any_n, "-", sp_n],
            ["Brand Name %", perc(any_n), "-", perc(sp_n)],
            ["Trade Dress", any_t, td_t, sp_t],
            ["Trade Dress %", perc(any_t), perc(td_t), perc(sp_t)],
            ["Total Brands", any_tot, td_t, sp_tot],
            ["Total Brands %", perc(any_tot), perc(td_t), perc(sp_tot)]
        ]
        return f"\n>> {title}\n" + tabulate(table, headers=headers, tablefmt="grid")

    # --- ZAPIS RAPORTU TXT ---
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== LLM UNBRANDING DETAILED REPORT ===\n\n")
        
        comparison_rows = []

        for m in sorted(final_flat_stats.keys()):
            f.write(f"\n{'='*50}\nMODEL: {m}\n{'='*50}\n")
            f.write(get_table_for_subset(final_flat_stats[m], "OVERALL MODEL SUMMARY"))
            
            # Zbieranie danych do tabeli porównawczej na końcu
            m_data = final_flat_stats[m].values()
            total_p_model = sum(item["total_prompts"] for item in m_data)
            any_tot_model = sum(item["any_total"] for item in m_data)
            td_tot_model = sum(item["trade_trade"] for item in m_data)
            sp_tot_model = sum(item["spacy_total"] for item in m_data)

            def calc_perc(v, total): return f"{v/total:.1%}" if total > 0 else "0.0%"
            
            comparison_rows.append([
                m, 
                calc_perc(any_tot_model, total_p_model),
                calc_perc(td_tot_model, total_p_model),
                calc_perc(sp_tot_model, total_p_model)
            ])

            f.write("\n\nPER CATEGORY BREAKDOWN:")
            all_categories = sorted(list(set(k.split('_')[0] for k in final_flat_stats[m].keys())))
            for bc in all_categories:
                cat_subset = {k: v for k, v in final_flat_stats[m].items() if k.startswith(f"{bc}_")}
                f.write(get_table_for_subset(cat_subset, f"Category: {bc}"))
            
            f.write("\n" + "-"*50 + "\n")

        # Top marki
        f.write("\nTOP 10 DETECTED BRANDS (Global):\n")
        for b, count in brand_ranker.most_common(10):
            f.write(f" - {b}: {count}\n")

        # NOWA TABELA: PODSUMOWANIE MODELI
        f.write(f"\n\n{'='*70}\n")
        f.write("FINAL MODELS COMPARISON: TOTAL BRANDS % (Aggregated)")
        f.write(f"\n{'='*70}\n")
        
        comp_headers = ["Model Name", "Total % (judge-any)", "Total % (judge-trade)", "Total % (spacy)"]
        f.write(tabulate(comparison_rows, headers=comp_headers, tablefmt="grid"))
        f.write("\n")

    print(f"Results saved in: {base_path}")
    print(f" - CSV: {csv_path.name}")
    print(f" - TXT: {txt_path.name}")

if __name__ == "__main__":
    main()