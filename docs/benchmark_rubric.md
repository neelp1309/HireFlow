# Senior Accountant Relevance Rubric

HireFlow was supplied with one Senior Accountant job description and 50 synthetic accounting/finance resumes, but no relevance labels. Phase 4 therefore introduces a **project-authored silver benchmark** so retrieval and reranking can be measured instead of judged only by visual inspection.

## Label scale

| Label | Name | Interpretation |
|---:|---|---|
| 3 | Strong fit | Direct accountant-family role, generally within the 3-5 year target, with broad evidence aligned to the Senior Accountant responsibilities. |
| 2 | Moderate fit | Meaningfully relevant, but has one notable mismatch such as adjacent role family, slightly low/high experience, specialization, or narrower evidence. |
| 1 | Weak fit | Accounting/finance relevance exists, but there are material role, seniority, experience, or breadth gaps. |
| 0 | Not suitable | Clear mismatch with the required level, especially recent-graduate/entry profiles without 3-5 years or executive-level profiles far beyond the target. |

For binary retrieval metrics, **labels 2 and 3 are treated as relevant**. NDCG uses the full 0-3 graded scale. `Strong@K` reports the proportion of label-3 candidates in the top K.

## Annotation principles

Labels were authored from the supplied JD and parsed resume evidence while avoiding candidate identity. The benchmark does **not** use candidate names, email addresses, phone numbers, gender, race, age, religion, marital status, photos, or other protected/sensitive attributes.

The annotation considers job-related evidence only:

- role/function alignment with Senior Accountant work;
- 3-5 years required experience;
- financial reporting and closing responsibilities;
- journal entries and reconciliations;
- budgeting/variance analysis;
- payroll, audit, tax, AP/AR and GAAP-related exposure;
- relevant accounting/ERP software;
- accounting/finance education;
- preferred credentials and leadership evidence as secondary factors.

## Important limitation

This is a **silver benchmark**, not independently adjudicated recruiter ground truth. It is useful for engineering iteration and portfolio demonstration, but production claims should use multiple human annotators, inter-rater agreement, adjudication, and multiple job descriptions.

The hybrid scoring weights are deliberately **not tuned against this one benchmark**. Optimizing them against one JD would overfit the evaluation set and make the reported metrics misleading.
