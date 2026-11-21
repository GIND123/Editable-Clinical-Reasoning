# Task 5: Validator adjudication

**Why it matters.** The validator flags clinician assertions that contradict the record. Some cases are genuinely ambiguous: the note itself may be wrong, findings may be documented inconsistently (the HPI says "fevers", the ROS says "negative for fever"), or a value may have changed over time. How we treat these determines the reported trade-off between catching false claims and over-flagging.

**What you receive.** A sample of (note excerpt, asserted edit, validator decision, validator reasons) for:
- flagged edits we labelled *valid* (potential false flags);
- unflagged edits we labelled *invalid* (potential misses).

**For each item**
- `correct_decision`: flag / accept / ambiguous.
- `reason`: documentation inconsistency / temporal change / validator error / gold label error / other.
- `comment`.

Items you mark *ambiguous* are reported separately and excluded from the headline sensitivity / false-flag numbers, with the exclusion rule pre-registered.
