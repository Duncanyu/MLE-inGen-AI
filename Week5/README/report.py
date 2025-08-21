import os, json
from pathlib import Path

IN_PATH  = "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week5/Homework/reports/generations.json"
OUT_PATH = "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week5/Homework/reports/report.md"

def main():
    os.makedirs("reports", exist_ok=True)
    data = json.loads(Path(IN_PATH).read_text(encoding="utf-8"))

    lines = []
    lines.append("# Week 5 SFT Report\n")
    lines.append("## Qualitative Comparison\n")
    for i, row in enumerate(data, 1):
        lines.append(f"### Prompt {i}\n")
        lines.append(f"**Prompt:** {row['prompt']}\n")
        lines.append(f"**Baseline:** {row['baseline'][:800]}{'...' if len(row['baseline'])>800 else ''}\n")
        lines.append(f"**LoRA:** {row['lora'][:800]}{'...' if len(row['lora'])>800 else ''}\n")
        if row.get("full"):
            lines.append(f"**Full FT:** {row['full'][:800]}{'...' if len(row['full'])>800 else ''}\n")
        lines.append("\n---\n")

    lines.append("## Summary Table (fill during evaluation)\n")
    lines.append("| Model | Notes (style/helpfulness/accuracy) |\n|---|---|\n")
    lines.append("| Baseline |  |\n| LoRA |  |\n| Full FT |  |\n")

    Path(OUT_PATH).write_text("\n".join(lines), encoding="utf-8")
    print(f"repoprted to {OUT_PATH}")

if __name__ == "__main__":
    main()
