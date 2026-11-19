# Results

The paper's numbers: one JSON file, a golden, per result, under its experiment's directory, as
`ruleforge reproduce all rebuild` writes them from the published campaign roots.

```bash
uv run ruleforge reproduce all rebuild --out DIR                  # recomputes them into DIR/results/
uv run ruleforge reproduce all verify --rerun DIR --exact         # compares that rebuild with them
uv run ruleforge reproduce all render --out FIGURES               # draws the paper's figures and tables from them
```

[docs/reproduce.md](../docs/reproduce.md#the-figures-and-tables) names the paper's figures and tables each file holds.
