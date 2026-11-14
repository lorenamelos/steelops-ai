# SteelOps AI

Industrial energy forecasting and decision support for a steel processing facility.

> **Status:** Milestone 1 (project foundation and data audit). No models have been trained yet, and no results are reported.

## Data

[Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption), UCI Machine Learning Repository, dataset 851 (DOI [10.24432/C52G8C](https://doi.org/10.24432/C52G8C)), CC BY 4.0. Sathishkumar V E, Shin C., Cho Y. (2021).

Raw files are never modified: they are checksummed (SHA-256 pinned in `configs/data.yaml`), recorded in `data/raw/MANIFEST.json`, and set read-only. Corrections are applied in code (`steelops.data.clean`), with tests.

## Reproduce

```bash
uv sync                          # install pinned dependencies (Python 3.12)
uv run steelops-download         # download the official UCI zip and verify checksums
uv run steelops-audit            # write the data-quality report to reports/data_audit.json
uv run pytest                    # run the test suite
uv run jupyter lab notebooks/    # open the exploratory notebook
```

If the UCI host is unreachable, download the official zip manually and run
`uv run steelops-download --from-zip path/to/steel+industry+energy+consumption.zip`.

## License

Code is released under the [MIT License](LICENSE). The dataset is not redistributed in this repository; it is downloaded from UCI and remains under its own license (CC BY 4.0), credited above.
