# Missing Callers Under a Response Budget

**Finishing Versus Finding in a Local Gemma 4 Agent**

Arjun Thilak R (AI Researcher, Eros Innovation) and Ramkumar MV (AI Researcher, Eros Innovation)

- Paper: [`paper/Missing_Callers_paper.pdf`](paper/Missing_Callers_paper.pdf)
- Resource writeup (OmitBench): [`paper/OmitBench_resource.pdf`](paper/OmitBench_resource.pdf)
- Dataset package (every episode, call-site truth, verifier): https://www.kaggle.com/datasets/rarjunthilak/absent-edges-expa-v1
- Notebook that re-runs the checks on a CPU: https://www.kaggle.com/code/rarjunthilak/missing-callers-reproduce-research-results
- Competition: https://www.kaggle.com/competitions/gemma-4-developer-agent-paper

## What this is

Code-graph tools give repository agents caller lists that are often incomplete. We test whether a local Gemma 4 31B
agent still finds a caller that its graph answer leaves out. Over 30 pinned Python libraries (750 targets, 11,250
primary episodes), the answer omits one caller whose call ran in the library's own tests and keeps every other entry
byte-identical. Under an 8-response limit, the incomplete list raised recovery of the omitted caller by 15.6 points over
no list (95% CI 11.5 to 19.8). Follow-ups show that the gain mostly comes from finishing: one line before the last
response added 22.5 points over a neutral line, while the missing caller still cost 10.5 points against the complete
list. Five more local models confirm the omission cost; the list's effect depends on whether the agent keeps searching.

## Repository layout

| Path | Contents |
|---|---|
| `absent_edges/probe/` | the probe: item build, harness, serving, scoring |
| `scripts/` | run, score, analysis, figure and release scripts |
| `tests/` | unit tests (synthetic fixtures only) |
| `manifests/expA/` | frozen manifests: library registry, items, dispatch, findability |
| `reports/` | frozen result files of every run (analysis JSON, figures, checksums) |
| `releases/` | frozen file lists of the code trees each run used |
| `paper/` | the two papers (PDF) and every figure (PNG, PDF, SVG, with the values each one plots) |
| `notebooks/` | Kaggle notebooks that re-run the checks from the dataset package |
| `licenses/`, `THIRD_PARTY_NOTICES.txt` | licences of the 30 libraries whose code excerpts appear in the data |

## Reproduce

1. Download the dataset package from Kaggle and unpack it.
2. From its root, run the verifier (Python 3.11+, standard library only, CPU, no network):

   ```
   python -I -B reproduce/verify.py --root . --jobs 8
   ```

   It re-runs every frozen analysis from the released scores and checks every hash chain down to the episode
   records (96 checks).
3. Unit tests of this repository: `python -m unittest discover -s tests -v`.
4. Figures: `python scripts/paper_figures.py --help` (needs matplotlib; the style and font are in
   `paper/figures/style/`).

## Licence

Apache-2.0 (see `LICENSE` and `NOTICE`). Code excerpts of the 30 libraries keep their own licences
(`THIRD_PARTY_NOTICES.txt`, `licenses/`).

## Citation

See `CITATION.cff`.
