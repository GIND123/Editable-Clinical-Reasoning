# Task 2: Silver-graph audit (100 graphs)

**Why it matters.** Most training graphs are produced automatically from published case reasoning. Your audit estimates their error rate. That estimate goes in the paper and sets whether the filters must be tightened.

**What you receive.** `clinician/review/silver_audit.csv`, a random 100 kept graphs. Each row shows the case id, the reference diagnosis and the graph in readable form (observations, rationales, diagnoses, must-not-miss).

**Rubric (one row per graph)**
| Column | Scale | Guidance |
|---|---|---|
| obs_verbatim_and_polarity_ok | 1–5 | Are the findings real, relevant and correctly marked present / absent? |
| rationales_correct | 1–5 | Are the medical claims correct, and do they link to the right diagnosis? |
| diagnosis_correct | Y/N | Is the most specific diagnosis correct for the case? |
| mnm_sensible | Y/N | Are must-not-miss statuses reasonable? |
| critical_error | Y/N | Anything that would teach the model something dangerous |
| comments | free text | Especially recurring error patterns |

**Effort.** About 3–5 minutes per graph. Please flag any systematic pattern early (e.g. negations lost), because a fix to the generator is cheaper than auditing more.
