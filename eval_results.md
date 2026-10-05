# Golden-set eval

Service: `/ask` (k=8), judge: gpt-4o-mini

| # | Question | Expected | Got | Cited | Retrieval hit (top-5) | Faithful | Correct | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | How many remote days are allowed? | Up to 3 days per week | Employees may work remotely up to three days per week. | POL-101 | ✅ | ✅ | ✅ | evidence at rank 2 |
| 2 | What is the mileage rate? | 45p per mile over 50 miles | The mileage rate is 45 pence per mile for travel over 50 miles. | POL-114 | ✅ | ✅ | ✅ | evidence at rank 1 |
| 3 | How quickly must a lost laptop be reported? | Within 1 hour | A lost laptop must be reported within one hour to the security desk. | POL-207 | ✅ | ✅ | ✅ | evidence at rank 1 |
| 4 | What is the WB-9 payload limit? | 25 kg | The WB-9 warehouse robot has a payload capacity limit of 25 kg. | SPEC-WB9 | ✅ | ✅ | ✅ | evidence at rank 1 |
| 5 | What are the company's values? | Safety first, Customer truth, Own the outcome, Teach what you learn | The company's values are: Safety first, Customer truth, Own the outcome, and Teach what you learn. | POL-101 | ✅ | ✅ | ✅ | evidence at rank 1 |
| 6 | How much can I claim for home office equipment? | Up to £350 every 36 months | Northwind reimburses up to £350 (or local equivalent) once every 36 months for approved home office equipment. | POL-101 | ✅ | ✅ | ✅ | evidence at rank 5 |
| 7 | What is the parental leave policy? | Refuse (not in docs) | I don't have enough information to answer that. | — | n/a | ✅ | ✅ | refused |
| 8 | Do employees get a company car? | Refuse (not in docs) | I don't have enough information to answer that. | — | n/a | ✅ | ✅ | refused |

Retrieval hit: 6/6 · Faithful: 8/8 · Correct (incl. refusals): 8/8
