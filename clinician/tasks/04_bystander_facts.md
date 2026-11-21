# Task 4: Bystander facts and edit wording

**Bystander facts** (`configs/benchmark/bystander_facts.yaml`) are inserted into cases to test that irrelevant information changes nothing. A fact is acceptable only if it is irrelevant to the diagnosis in **every** one of the 25 categories.

**Remove anything with plausible clinical meaning:**
- occupation, travel, animals and pets, hobbies (gardening, for example);
- sick contacts, family illness, diet, substances, medications;
- housing, sexual history, recent procedures.

**Add** neutral administrative or logistical facts if the list feels thin; aim for about 20.

**Edit-wording templates** (`src/clinblueprint/edits/text_adapter.py`) render edits into free text for the CoT and note-level comparisons. Examples:
- "Suppose that there is no productive cough."
- "Clinician correction: the finding '…' is equivocal and of limited weight."

Please flag any wording a clinician would find unnatural or ambiguous.

When done, set `status: APPROVED` in the YAML.
