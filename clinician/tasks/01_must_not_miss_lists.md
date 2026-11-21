# Task 1: Must-not-miss lists (25 DiReCT categories)

**Why it matters.** For each disease category the model works in, it must explicitly address the dangerous alternatives on this list, marking each excluded, open or not addressed. Missing items are inserted automatically as *not addressed*, which makes omissions visible to the user and to our metrics.

**What to do**
1. Open `configs/schema/must_not_miss.yaml`. It is a **DRAFT** written as a starting point.
2. For each category, keep, remove or add diagnoses. Aim for conditions that are (a) plausible in that presentation, (b) dangerous if missed and (c) actionable. Typically 3–6 per category.
3. Use the names you would write in a note. Synonyms are fine; we normalise them.
4. When done, set `status: APPROVED`, write your initials in `approved_by` and increment `version`.

**Questions to keep in mind**
- Is anything on the list so rare in that presentation that listing it would add noise?
- Are there category-specific safety issues the draft missed? Examples: pre-excited AF before AV-nodal blockade; hypoglycaemia mimicking stroke.
