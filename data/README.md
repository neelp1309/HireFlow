# Data layout

Raw candidate resumes and job descriptions are intentionally **not committed** to Git.
Even when a dataset is synthetic, a resume corpus contains identity-like fields and should
be treated as sensitive by default.

Place local input files as follows:

```text
data/
├── raw/
│   ├── resumes/    # Resume PDFs
│   └── jobs/       # Job-description PDFs
├── processed/      # Generated normalized JSON/Parquet
└── evaluation/     # Ground-truth labels and experiment fixtures
```

For the supplied HireFlow dataset, copy the 50 PDFs from `Resume_dataset.zip` into
`data/raw/resumes/` and copy `Senior_Accountant_Position.pdf` into `data/raw/jobs/`.
