from pathlib import Path
import os, json

BASE_DIR    = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
GENS_PATH   = REPORTS_DIR / "generations.json"
OUT_PATH    = REPORTS_DIR / "report.md"

def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    data = json.loads(GENS_PATH.read_text(encoding="utf-8"))

    lines = ["# Week 5 SFT Report\n", "## Qualitative Comparison\n"]
    for i, row in enumerate(data, 1):
        lines += [
            f"### Prompt {i}\n",
            f"**Prompt:** {row['prompt']}\n",
            f"**Baseline:** {row['baseline'][:800]}{'...' if len(row['baseline'])>800 else ''}\n",
            f"**LoRA:** {row['lora'][:800]}{'...' if len(row['lora'])>800 else ''}\n",
        ]
        if row.get("full"):
            lines.append(f"**Full FT:** {row['full'][:800]}{'...' if len(row['full'])>800 else ''}\n")
        lines.append("\n---\n")

    lines += [
        "## Summary Table (fill during evaluation)\n",
        "| Model | Notes (style/helpfulness/accuracy) |\n|---|---|\n",
        "| Baseline |  |\n| LoRA |  |\n| Full FT |  |\n"
    ]

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"repoprted to {OUT_PATH}")

if __name__ == "__main__":
    main()
