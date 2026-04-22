## 🏆 Indic Language Benchmark Leaderboard

### Overall Results (Primary Metric per Task)

| Model     |   code_mixed_qa |   math_reasoning |   reading_comprehension |   Average |
|:----------|----------------:|-----------------:|------------------------:|----------:|
| Gemma-2B  |            8.56 |               10 |                   27.43 |     15.33 |
| Sarvam-2B |           17.46 |               14 |                    1.87 |     11.11 |

### Code Mixed Qa

| Model     | Task          | Language   |   EM (%) |   F1 (%) | Accuracy (%)   |   BLEU (%) |
|:----------|:--------------|:-----------|---------:|---------:|:---------------|-----------:|
| Gemma-2B  | code_mixed_qa | Overall    |        0 |     8.56 | -              |       4.86 |
| Sarvam-2B | code_mixed_qa | Overall    |        4 |    17.46 | -              |       6.49 |

### Math Reasoning

| Model     | Task           | Language   | EM (%)   | F1 (%)   |   Accuracy (%) | BLEU (%)   |
|:----------|:---------------|:-----------|:---------|:---------|---------------:|:-----------|
| Gemma-2B  | math_reasoning | Overall    | -        | -        |             10 | -          |
| Gemma-2B  | math_reasoning | hindi      | -        | -        |             10 | -          |
| Gemma-2B  | math_reasoning | bengali    | -        | -        |              8 | -          |
| Gemma-2B  | math_reasoning | tamil      | -        | -        |             12 | -          |
| Sarvam-2B | math_reasoning | Overall    | -        | -        |             14 | -          |
| Sarvam-2B | math_reasoning | hindi      | -        | -        |             12 | -          |
| Sarvam-2B | math_reasoning | bengali    | -        | -        |             16 | -          |
| Sarvam-2B | math_reasoning | tamil      | -        | -        |             14 | -          |

### Reading Comprehension

| Model     | Task                  | Language   |   EM (%) |   F1 (%) | Accuracy (%)   | BLEU (%)   |
|:----------|:----------------------|:-----------|---------:|---------:|:---------------|:-----------|
| Gemma-2B  | reading_comprehension | Overall    |    17.33 |    27.43 | -              | -          |
| Gemma-2B  | reading_comprehension | bengali    |     8    |    20.16 | -              | -          |
| Gemma-2B  | reading_comprehension | hindi      |    32    |    39.41 | -              | -          |
| Gemma-2B  | reading_comprehension | tamil      |    12    |    22.7  | -              | -          |
| Sarvam-2B | reading_comprehension | Overall    |     0    |     1.87 | -              | -          |
| Sarvam-2B | reading_comprehension | bengali    |     0    |     0.48 | -              | -          |
| Sarvam-2B | reading_comprehension | hindi      |     0    |     2.92 | -              | -          |
| Sarvam-2B | reading_comprehension | tamil      |     0    |     2.22 | -              | -          |
