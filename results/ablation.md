
## All sets (hit@5 / hit@3 / MRR); n = dev 20, test 50, hard 46

| mode | dev | test | hard |
|---|---|---|---|
| dense_whole | 95% / 95% / 0.92 | 100% / 100% / 0.98 | 89% / 70% / 0.76 |
| dense | 100% / 100% / 0.94 | 100% / 100% / 0.98 | 93% / 80% / 0.79 |
| bm25 | 90% / 90% / 0.76 | 100% / 100% / 0.94 | 70% / 57% / 0.73 |
| hybrid | 95% / 95% / 0.95 | 100% / 100% / 0.98 | 91% / 85% / 0.81 |
| dense_x | 100% / 100% / 0.94 | 100% / 100% / 0.98 | 93% / 85% / 0.80 |
| hybrid_x | 80% / 65% / 0.68 | 90% / 80% / 0.80 | 85% / 74% / 0.76 |
| dense+rerank | 100% / 100% / 1.00 | 100% / 100% / 0.99 | 98% / 93% / 0.91 |
| hybrid+rerank | 95% / 95% / 0.95 | 100% / 100% / 0.99 | 91% / 89% / 0.90 |

## Hard set by category (hit@5)

| mode | buried (n=12) | colloquial (n=8) | lifecycle (n=10) | mixed (n=6) | multi (n=10) | en | ar |
|---|---|---|---|---|---|---|---|
| dense_whole | 100% | 75% | 100% | 83% | 80% | 85% | 92% |
| dense | 100% | 88% | 100% | 100% | 80% | 90% | 96% |
| bm25 | 92% | 12% | 100% | 67% | 60% | 65% | 73% |
| hybrid | 100% | 75% | 100% | 100% | 80% | 90% | 92% |
| dense_x | 100% | 75% | 100% | 100% | 90% | 90% | 96% |
| hybrid_x | 83% | 75% | 100% | 100% | 70% | 70% | 96% |
| dense+rerank | 100% | 100% | 100% | 100% | 90% | 95% | 100% |
| hybrid+rerank | 100% | 75% | 100% | 100% | 80% | 90% | 92% |

## Hard set: share of questions with ALL needed controls in the top k

| mode | k=1 | k=3 | k=5 | k=8 | k=10 |
|---|---|---|---|---|---|
| dense_whole | 43% | 70% | 89% | 96% | 98% |
| dense | 43% | 80% | 93% | 96% | 100% |
| bm25 | 43% | 57% | 70% | 80% | 85% |
| hybrid | 48% | 85% | 91% | 91% | 91% |
| dense_x | 46% | 85% | 93% | 96% | 100% |
| hybrid_x | 46% | 74% | 85% | 89% | 89% |
| dense+rerank | 61% | 93% | 98% | 100% | 100% |
| hybrid+rerank | 61% | 89% | 91% | 91% | 91% |

## Paired comparison against dense (hit@5, all sets together)

- dense_whole: wins 0 [] | losses 3 ['dev/en', 'hard/h-C4-en', 'hard/h-X6-ar']
- bm25: wins 1 ['hard/h-M4-en'] | losses 14 ['dev/en', 'dev/en', 'hard/h-M2-en', 'hard/h-M2-ar', 'hard/h-M3-ar', 'hard/h-B5-en', 'hard/h-C1-en', 'hard/h-C1-ar', 'hard/h-C2-en', 'hard/h-C3-en', 'hard/h-C3-ar', 'hard/h-C4-en', 'hard/h-X4-ar', 'hard/h-X5-ar']
- hybrid: wins 1 ['hard/h-M4-en'] | losses 3 ['dev/en', 'hard/h-M3-ar', 'hard/h-C3-en']
- dense_x: wins 2 ['hard/h-M3-en', 'hard/h-M4-en'] | losses 2 ['hard/h-M2-en', 'hard/h-C3-en']
- hybrid_x: wins 0 [] | losses 13 ['dev/en', 'dev/en', 'dev/en', 'dev/en', 'test/en-03', 'test/en-05', 'test/en-08', 'test/en-10', 'test/en-11', 'hard/h-M2-en', 'hard/h-B3-en', 'hard/h-B5-en', 'hard/h-C3-en']
- dense+rerank: wins 2 ['hard/h-M4-en', 'hard/h-C2-ar'] | losses 0 []
- hybrid+rerank: wins 1 ['hard/h-M4-en'] | losses 3 ['dev/en', 'hard/h-M3-ar', 'hard/h-C3-en']

## Questions dense gets wrong (top 5)

- [hard] h-M3-en wanted ['1-9-5', '2-6-3'] ranks {'1-9-5': 7, '2-6-3': 1} got ['2-6-3', '4-1-2', '1-9-3', '1-9-1', '1-9-4']
- [hard] h-M4-en wanted ['2-13-3', '3-1-3'] ranks {'2-13-3': 9, '3-1-3': 1} got ['3-1-3', '2-9-4', '2-9-1', '2-9-3', '3-1-1']
- [hard] h-C2-ar wanted ['3-1-3'] ranks {'3-1-3': 10} got ['2-6-3', '2-3-3', '2-9-3', '2-12-3', '2-5-3']

## Questions dense+rerank gets wrong (top 5)

- [hard] h-M3-en wanted ['1-9-5', '2-6-3'] ranks {'1-9-5': 7, '2-6-3': 1} got ['2-6-3', '2-14-3', '2-14-2', '1-9-1', '4-1-2']

saved ablation.json
