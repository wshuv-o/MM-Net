# Project page

Served at https://wshuv-o.github.io/MM-Net by GitHub Pages from this folder.

`index.html` reads everything it displays from `data/`, and those files are
generated from the result files in `results/` rather than typed in. So the page
cannot quietly drift away from the paper: regenerate the data and the page follows.

| file | generated from |
|---|---|
| `data/results.json`     | `results/revision/runs/final/derived_seed42.json`, `fold_level_tests.json` |
| `data/patients.json`    | `results/revision/runs/final/per_subject_seed42.json` plus the clinical AHI table |
| `data/benchmark.json`   | the staging benchmark table, parsed straight out of the manuscript source |
| `data/checkpoints.json` | `results/checkpoints/manifest.json` |

The checkpoints themselves are attached to a GitHub release rather than committed,
so a clone stays small. The page links to the release and lists what is in it.

No PDF and no manuscript text is hosted here while the paper is under review.
