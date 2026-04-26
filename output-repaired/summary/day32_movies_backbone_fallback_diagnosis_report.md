# Day32 Movies SASRec Fallback Diagnosis

## 1. Day31 Recap

Day31 established a positive cross-domain CEP calibration result on Movies medium_5neg_2000: valid/test inference completed with parse_success=1.0, and raw relevance ECE dropped from 0.2913 to calibrated ECE 0.0044 on test. However, the Movies SASRec plug-in was not healthy because fallback_rate was 0.9698.

## 2. Fallback Breakdown

Rows: `12000`. Fallback rows: `11637` (`0.9698`). Positive fallback rate is `0.8965` and negative fallback rate is `0.9844`.

Fallback is driven by two issues:

- User/history mapping: `4122` fallback rows also have missing/unmapped train history under the original title-based mapper.
- Candidate item coldness: `11137` fallback rows have candidate items absent from the train SASRec vocabulary.
- Both conditions overlap on `3899` rows.

## 3. ID / Schema Consistency

Movies medium stores `candidate_item_id` as raw ASIN but many `history` entries as text such as `Item ID: B07...`. This is schema-compatible for prompting, but it is not ideal for the Beauty SASRec helper, which originally maps history through title strings. A robust ASIN extractor fixes the history-format mismatch without changing the original data files.

## 4. Split / Vocab Audit

Test users with train history after robust extraction: `1296` / `2000`. Test candidate items in train vocabulary: `644` / `11480`. Test candidate cold rate: `0.9439`. Test positive cold rate: `0.7876`.

This means the main remaining blocker is not merely ID dtype. Movies medium has substantial future/candidate coldness relative to the train split, including many positive test items that are not in the train SASRec item vocabulary.

## 5. Repair Attempts

The non-invasive repair ablation shows the best fallback rate remains `0.9307` under `id_repaired`. This does not meet the `<20%` health threshold. Adding candidate-only items to the scoring vocabulary would not train their embeddings without using test labels; therefore, cold candidates must still use an explicit fallback such as train popularity or min score.

## 6. Decision

Fallback could not be reduced below 20% without changing the data split or training objective. Therefore, Day32 does not rerun Movies SASRec plug-in as a healthy external-backbone result.

## 7. Next Step

Recommended Day33 path: either move to Books medium5 and check whether its sequential backbone coverage is healthier, or try a Movies backbone less sensitive to cold candidates, such as GRU4Rec/Bert4Rec from the LLM-ESR path. The Movies CEP calibration result remains valid; only the Movies SASRec plug-in claim is blocked.
