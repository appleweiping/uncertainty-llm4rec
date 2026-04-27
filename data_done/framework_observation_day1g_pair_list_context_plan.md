# Framework-Observation-Day1g Pair/List Context Plan

If pointwise logit P(true) and Day1f self-consistency remain weak, the next step is not more scalar confidence wording. The input context should change.

## Pairwise Context

Give the model one user history plus two candidates, ideally one positive-like and one negative-like, and ask which candidate better matches the user. Confidence can be extracted from pairwise preference margins or calibrated pairwise probabilities.

## Listwise Context

Give the model the full six-candidate user pool and ask for a ranking or relative score for each candidate. This matches the recommendation task more directly than isolated pointwise decisions.

## Confidence Extraction

Use relative decision margins, rank gaps, calibrated probabilities, or consistency across listwise rankings. Do not return to raw verbalized scalar confidence unless it is only used as a secondary diagnostic.

## Scope

Day1g should remain Beauty-only smoke first. Do not run full Beauty or four domains until pair/list context clearly beats pointwise logit and self-consistency on the same user-pool ranking metrics.
