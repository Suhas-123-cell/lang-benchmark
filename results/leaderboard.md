## 🏆 Indic Language Benchmark Leaderboard

### Overall Results (Primary Metric per Task)

| Model     |   code_mixed_qa |   math_reasoning |   reading_comprehension |   Average |
|:----------|----------------:|-----------------:|------------------------:|----------:|
| Gemma-2B  |            8.56 |             8.33 |                   30.63 |     15.84 |
| Sarvam-2B |           17.46 |            13.33 |                    1.89 |     10.89 |

### Code Mixed Qa

| Model     | Task          | Language   |   EM (%) |   F1 (%) | Accuracy (%)   |   BLEU (%) |
|:----------|:--------------|:-----------|---------:|---------:|:---------------|-----------:|
| Gemma-2B  | code_mixed_qa | Overall    |        0 |     8.56 | -              |       4.86 |
| Sarvam-2B | code_mixed_qa | Overall    |        4 |    17.46 | -              |       6.49 |

### Math Reasoning

| Model     | Task           | Language   | EM (%)   | F1 (%)   |   Accuracy (%) | BLEU (%)   |
|:----------|:---------------|:-----------|:---------|:---------|---------------:|:-----------|
| Gemma-2B  | math_reasoning | Overall    | -        | -        |           8.33 | -          |
| Gemma-2B  | math_reasoning | hindi      | -        | -        |           9    | -          |
| Gemma-2B  | math_reasoning | bengali    | -        | -        |           8    | -          |
| Gemma-2B  | math_reasoning | tamil      | -        | -        |           8    | -          |
| Sarvam-2B | math_reasoning | Overall    | -        | -        |          13.33 | -          |
| Sarvam-2B | math_reasoning | hindi      | -        | -        |          13    | -          |
| Sarvam-2B | math_reasoning | bengali    | -        | -        |          13    | -          |
| Sarvam-2B | math_reasoning | tamil      | -        | -        |          14    | -          |

### Reading Comprehension

| Model     | Task                  | Language   |   EM (%) |   F1 (%) | Accuracy (%)   | BLEU (%)   |
|:----------|:----------------------|:-----------|---------:|---------:|:---------------|:-----------|
| Gemma-2B  | reading_comprehension | Overall    |    20.67 |    30.63 | -              | -          |
| Gemma-2B  | reading_comprehension | bengali    |    12    |    21.75 | -              | -          |
| Gemma-2B  | reading_comprehension | hindi      |    35    |    43.55 | -              | -          |
| Gemma-2B  | reading_comprehension | tamil      |    15    |    26.6  | -              | -          |
| Sarvam-2B | reading_comprehension | Overall    |     0    |     1.89 | -              | -          |
| Sarvam-2B | reading_comprehension | bengali    |     0    |     0.32 | -              | -          |
| Sarvam-2B | reading_comprehension | hindi      |     0    |     2.28 | -              | -          |
| Sarvam-2B | reading_comprehension | tamil      |     0    |     3.07 | -              | -          |
