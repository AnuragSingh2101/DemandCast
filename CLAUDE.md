# DemandCast - Standing Rules (from Master Implementation Prompt Section 2)

These rules must be obeyed in every phase of the project.

## 1. No Leakage
Every feature uses only information available at the forecast origin. No rolling statistic may touch a future observation.

## 2. No Fabrication
No fake demand, no hardcoded results, no invented KPIs, no performance numbers that were not produced by a run you executed. Never claim something works unless you ran it.

## 3. Honest Model Selection
Choose winners from out-of-sample results only. A baseline winning is an acceptable and reportable outcome. Do not tune until a complex model wins. Never hardcode a winner.

## 4. Chronological Validation Only
No random train/test splits anywhere.

## 5. Privacy
Drop customer email, names, password, street address, and unneeded customer/order identifiers during ETL. Never print or display them in notebooks, logs, dashboard tables, or reports.

## 6. Evidence Before Assumptions
Never assume a column exists or a date format is consistent. Inspect first, then document every assumption.

## 7. Windows First
All commands must work in PowerShell. Never give bash-only instructions.

## 8. Continuity
Before starting any work, read docs/IMPLEMENTATION_STATUS.md if it exists. Update it after every phase. Do not redo finished work or overwrite working files blindly.

## 9. Execute and Verify
Run what you write. Read the outputs. Run the tests. Fix failures before moving on.

## 10. Keep It CPU-Friendly and Free
No paid APIs, no cloud requirement, no GPU-dependent models, no unnecessary containers or technologies.