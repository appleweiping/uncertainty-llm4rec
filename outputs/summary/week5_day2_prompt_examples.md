# Week5 Day2 Prompt and Parser Examples

## candidate_ranking | json_full | success=True

Prompt:
```text
You are a recommendation ranking assistant.

The user has interacted with the following items:
1. OZNaturals Retinol Serum

You must rank only the following 6 candidate items. Do not invent new items and do not use any item outside this list.

Candidates:
1. item_id=B01JMRYD8Q
   title=Retinol Face Masks: WHOLESALE Hydrogel Sheet Masks with Plant Collagen, Vitamin A, and Hydrating Ingredients. Reduce Redness & Brighten Skin while Combating Forehead and Mouth Wrinkles (4 MASKS)
   text=Title: Retinol Face Masks: WHOLESALE Hydrogel Sheet Masks with Plant Collagen, Vitamin A, and Hydrating Ingredients. Reduce Redness & Brighten Skin while Combating Forehead and Mouth Wrinkles (4 MASKS)
   popularity_group=tail
2. item_id=B00OS9YWJY
   title=Pure Hyaluronic Acid Serum with Vitamin C for Face - Organic Anti Aging & Anti-Wrinkle Skin Treatment - Penetrates Deeply & Activates Collagen Production for a Younger, Firmer & Brighter Complexion
   text=Title: Pure Hyaluronic Acid Serum with Vitamin C for Face - Organic Anti Aging & Anti-Wrinkle Skin Treatment - Penetrates Deeply & Activates Collagen Production for a Younger, Firmer & Brighter Complexion
   popularity_group=mid
3. item_id=B09PNG
...[truncated]...
```

Response:
```text
{
  "ranked_item_ids": [
    "B00OS9YWJY",
    "B01JMRYD8Q",
    "B09PNGDR81",
    "B08CXWTK1G",
    "B07RBSLNFR",
    "B0876L7BWM"
  ],
  "topk_item_ids": [
    "B00OS9YWJY",
    "B01JMRYD8Q",
    "B09PNGDR81"
  ],
  "confidence": 0.83,
  "reason": "history aligns most strongly with the first ranked candidate."
}
```

Parsed:
```json
{
  "ranked_item_ids": [
    "B00OS9YWJY",
    "B01JMRYD8Q",
    "B09PNGDR81",
    "B08CXWTK1G",
    "B07RBSLNFR",
    "B0876L7BWM"
  ],
  "topk_item_ids": [
    "B00OS9YWJY",
    "B01JMRYD8Q",
    "B09PNGDR81"
  ],
  "confidence": 0.83,
  "reason": "history aligns most strongly with the first ranked candidate.",
  "parse_mode": "json",
  "parse_success": true,
  "out_of_candidate_item_ids": [],
  "contains_out_of_candidate_item": false
}
```

Repair note: Ranking parser should accept strict JSON, fenced JSON, and top-k-only outputs while explicitly rejecting out-of-candidate drift.

## candidate_ranking | topk_only | success=True

Prompt:
```text
You are a recommendation ranking assistant.

The user has interacted with the following items:
1. EZ Detangler Brush, Morgles 2 Pack Detangling Brush for Curly Hair Ez Hair Detangling Brush for Afro America/ African Hair 3a to 4c Wavy Hair

You must rank only the following 6 candidate items. Do not invent new items and do not use any item outside this list.

Candidates:
1. item_id=B08DX9P6V1
   title=Laloirelle Age Renewal Serum - Phyto-Biotics Stem Cells, Botanical Hyaluronic Acid, Retinol, Multi-Vitamin, Herbal Infusion - Organic serum for anti-aging, hydration, wrinkles, all skin types (1oz)
   text=Title: Laloirelle Age Renewal Serum - Phyto-Biotics Stem Cells, Botanical Hyaluronic Acid, Retinol, Multi-Vitamin, Herbal Infusion - Organic serum for anti-aging, hydration, wrinkles, all skin types (1oz)
   popularity_group=tail
2. item_id=B08J7W1VQL
   title=KOMOREBI 235PCS Temporary Tattoos 24 Sheets Body Stickers for Woman and Girls,Size Small Waterproof Fake Tattoos, Butterflies Style Body Art L5.9''xW4.1'' Tattoo for Party
   text=Title: KOMOREBI 235PCS Temporary Tattoos 24 Sheets Body Stickers for Woman and Girls,Size Small Waterproof Fake Tattoos, Butterflies Style Body Art L
...[truncated]...
```

Response:
```text
{
  "topk_item_ids": [
    "B07PZ8JNWJ",
    "B08DX9P6V1",
    "B08J7W1VQL"
  ],
  "confidence": "0.77",
  "reason": "these are the strongest items inside the candidate set."
}
```

Parsed:
```json
{
  "ranked_item_ids": [
    "B07PZ8JNWJ",
    "B08DX9P6V1",
    "B08J7W1VQL"
  ],
  "topk_item_ids": [
    "B07PZ8JNWJ",
    "B08DX9P6V1",
    "B08J7W1VQL"
  ],
  "confidence": 0.77,
  "reason": "these are the strongest items inside the candidate set.",
  "parse_mode": "text_mentions",
  "parse_success": true,
  "out_of_candidate_item_ids": [],
  "contains_out_of_candidate_item": false
}
```

Repair note: Ranking parser should accept strict JSON, fenced JSON, and top-k-only outputs while explicitly rejecting out-of-candidate drift.

## candidate_ranking | invalid_ooc | success=False

Prompt:
```text
You are a recommendation ranking assistant.

The user has interacted with the following items:
1. Women's Pleated Scarves Pashmina Shawl Wraps for Wedding Bride Bridesmaid Gifts (Black)
2. YsesoAi 28 Pcs Hair Curlers Spiral Curls No Heat Wave Hair Curlers Styling Kit Spiral Hair Curlers Magic Hair Rollers with 2 Pieces Styling Hooks for Most Kinds of Hairstyles (20 cm/ 7.8 inches)

You must rank only the following 6 candidate items. Do not invent new items and do not use any item outside this list.

Candidates:
1. item_id=B087J3H22J
   title=NEW Challenger Men's Rinse Free Hand Wash - 1oz - Portable, 30% Alcohol Hand Wash both Cleans & Moisturizes. Hydrates Hands, Forearms, & Skin - Cleansing 30% Alc. Based Formula Washes and Nourishes
   text=Title: NEW Challenger Men's Rinse Free Hand Wash - 1oz - Portable, 30% Alcohol Hand Wash both Cleans & Moisturizes. Hydrates Hands, Forearms, & Skin - Cleansing 30% Alc. Based Formula Washes and Nourishes
   popularity_group=head
2. item_id=B07XP7L6XC
   title=16 Jars Chrome Nail Powder Kit Mirror Effect Manicure Decoration Pigment 16 Colors Art Nail Powders with 16pcs Eyeshadow Sticks, 1g/Jar (Mirror-16 Jars)
   text=Title: 16 Jars Chrome Na
...[truncated]...
```

Response:
```text
{
  "ranked_item_ids": [
    "OUT_OF_SET_ITEM",
    "B08HMVTVZB"
  ],
  "topk_item_ids": [
    "OUT_OF_SET_ITEM"
  ],
  "confidence": 0.6,
  "reason": "the model drifted outside the candidate set."
}
```

Parsed:
```json
{
  "ranked_item_ids": [
    "OUT_OF_SET_ITEM",
    "B08HMVTVZB"
  ],
  "topk_item_ids": [
    "OUT_OF_SET_ITEM"
  ],
  "confidence": 0.6,
  "reason": "the model drifted outside the candidate set.",
  "parse_mode": "json",
  "parse_success": false,
  "out_of_candidate_item_ids": [
    "OUT_OF_SET_ITEM"
  ],
  "contains_out_of_candidate_item": true
}
```

Repair note: Ranking parser should accept strict JSON, fenced JSON, and top-k-only outputs while explicitly rejecting out-of-candidate drift.

## pairwise_preference | json_item_id | success=True

Prompt:
```text
You are a recommendation preference assistant.

The user has interacted with the following items:
1. OZNaturals Retinol Serum

Now compare the following two candidate items for this user.

Candidate A:
item_id=B01JMRYD8Q
title=Retinol Face Masks: WHOLESALE Hydrogel Sheet Masks with Plant Collagen, Vitamin A, and Hydrating Ingredients. Reduce Redness & Brighten Skin while Combating Forehead and Mouth Wrinkles (4 MASKS)
text=Title: Retinol Face Masks: WHOLESALE Hydrogel Sheet Masks with Plant Collagen, Vitamin A, and Hydrating Ingredients. Reduce Redness & Brighten Skin while Combating Forehead and Mouth Wrinkles (4 MASKS)

Candidate B:
item_id=B00OS9YWJY
title=Pure Hyaluronic Acid Serum with Vitamin C for Face - Organic Anti Aging & Anti-Wrinkle Skin Treatment - Penetrates Deeply & Activates Collagen Production for a Younger, Firmer & Brighter Complexion
text=Title: Pure Hyaluronic Acid Serum with Vitamin C for Face - Organic Anti Aging & Anti-Wrinkle Skin Treatment - Penetrates Deeply & Activates Collagen Production for a Younger, Firmer & Brighter Complexion

Task:
Decide which item is more preferable for this user. You must choose only one of the two given items.

Do not wrap the
...[truncated]...
```

Response:
```text
{
  "preferred_item": "B00OS9YWJY",
  "confidence": 0.81,
  "reason": "this item better aligns with the recent history."
}
```

Parsed:
```json
{
  "preferred_item": "B00OS9YWJY",
  "confidence": 0.81,
  "reason": "this item better aligns with the recent history.",
  "parse_mode": "json",
  "parse_success": true,
  "ambiguous_preference": false
}
```

Repair note: Pairwise parser should handle explicit item ids, fenced A/B shorthand, and free-form preference text while preserving an explicit ambiguous failure case.

## pairwise_preference | json_fenced_ab | success=True

Prompt:
```text
You are a recommendation preference assistant.

The user has interacted with the following items:
1. Women's Pleated Scarves Pashmina Shawl Wraps for Wedding Bride Bridesmaid Gifts (Black)
2. Laloirelle Age Defy Face Oil - Virgin Marula Oil, Rosehip Oil Infused with Organic Reishi Mushroom, Ginseng, Cordyceps Mushroom - Organic Facial Oil for Anti Aging, Wrinkles, Acne Scars, Dry Skin (1oz)
3. Laloirelle Age Renewal Serum - Phyto-Biotics Stem Cells, Botanical Hyaluronic Acid, Retinol, Multi-Vitamin, Herbal Infusion - Organic serum for anti-aging, hydration, wrinkles, all skin types (1oz)
4. Kidskin- Foaming Body Wash with Tea Tree Oil- Helps Combat Body and Foot Odor and Stubborn Body Acne and Skin Issues - Cruelty Free, Gluten Free, Paraben Free, Sulfate Free, Unscented, and Vegan- Made in USA
5. O!GETi Vitamin C Foam Cleanser | Foaming Facial Cleanser, Perfect Pore Cleansing Foam, Face Wash for Dry & Sensitive Skin, Sebum Reducing, Moisture Control & Minimizing Pores, Korean Skin Care, Gift for Women, Paraben-free, 4.23 Oz (120g)
6. SENGTERM Satin Bonnet Adjustable Sleep cap Double Layered Hair Bonnet Silky Cap for Women Curly Natural Long Hair (L, Black)
7. Cordless Water Flosse
...[truncated]...
```

Response:
```text
```json
{
  "preferred_item": "B",
  "confidence": "81%",
  "reason": "the chosen side is more compatible with the user trajectory."
}
```
```

Parsed:
```json
{
  "preferred_item": "B083QP6MX4",
  "confidence": 0.81,
  "reason": "the chosen side is more compatible with the user trajectory.",
  "parse_mode": "json",
  "parse_success": true,
  "ambiguous_preference": false
}
```

Repair note: Pairwise parser should handle explicit item ids, fenced A/B shorthand, and free-form preference text while preserving an explicit ambiguous failure case.

## pairwise_preference | natural_language | success=True

Prompt:
```text
You are a recommendation preference assistant.

The user has interacted with the following items:
1. Women's Pleated Scarves Pashmina Shawl Wraps for Wedding Bride Bridesmaid Gifts (Black)
2. Laloirelle Age Defy Face Oil - Virgin Marula Oil, Rosehip Oil Infused with Organic Reishi Mushroom, Ginseng, Cordyceps Mushroom - Organic Facial Oil for Anti Aging, Wrinkles, Acne Scars, Dry Skin (1oz)
3. Laloirelle Age Renewal Serum - Phyto-Biotics Stem Cells, Botanical Hyaluronic Acid, Retinol, Multi-Vitamin, Herbal Infusion - Organic serum for anti-aging, hydration, wrinkles, all skin types (1oz)
4. Kidskin- Foaming Body Wash with Tea Tree Oil- Helps Combat Body and Foot Odor and Stubborn Body Acne and Skin Issues - Cruelty Free, Gluten Free, Paraben Free, Sulfate Free, Unscented, and Vegan- Made in USA
5. O!GETi Vitamin C Foam Cleanser | Foaming Facial Cleanser, Perfect Pore Cleansing Foam, Face Wash for Dry & Sensitive Skin, Sebum Reducing, Moisture Control & Minimizing Pores, Korean Skin Care, Gift for Women, Paraben-free, 4.23 Oz (120g)
6. SENGTERM Satin Bonnet Adjustable Sleep cap Double Layered Hair Bonnet Silky Cap for Women Curly Natural Long Hair (L, Black)
7. Cordless Water Flosse
...[truncated]...
```

Response:
```text
I would choose B083QP6MX4 over B08RYQXJ88 for this user. Confidence: 0.73. Reason: the selected candidate is the safer preference choice.
```

Parsed:
```json
{
  "preferred_item": "B083QP6MX4",
  "confidence": 0.73,
  "reason": "the selected candidate is the safer preference choice.",
  "parse_mode": "text_preference",
  "parse_success": true,
  "ambiguous_preference": false
}
```

Repair note: Pairwise parser should handle explicit item ids, fenced A/B shorthand, and free-form preference text while preserving an explicit ambiguous failure case.

## pairwise_preference | ambiguous_invalid | success=False

Prompt:
```text
You are a recommendation preference assistant.

The user has interacted with the following items:
1. Women's Pleated Scarves Pashmina Shawl Wraps for Wedding Bride Bridesmaid Gifts (Black)
2. Laloirelle Age Defy Face Oil - Virgin Marula Oil, Rosehip Oil Infused with Organic Reishi Mushroom, Ginseng, Cordyceps Mushroom - Organic Facial Oil for Anti Aging, Wrinkles, Acne Scars, Dry Skin (1oz)
3. Laloirelle Age Renewal Serum - Phyto-Biotics Stem Cells, Botanical Hyaluronic Acid, Retinol, Multi-Vitamin, Herbal Infusion - Organic serum for anti-aging, hydration, wrinkles, all skin types (1oz)
4. Kidskin- Foaming Body Wash with Tea Tree Oil- Helps Combat Body and Foot Odor and Stubborn Body Acne and Skin Issues - Cruelty Free, Gluten Free, Paraben Free, Sulfate Free, Unscented, and Vegan- Made in USA
5. O!GETi Vitamin C Foam Cleanser | Foaming Facial Cleanser, Perfect Pore Cleansing Foam, Face Wash for Dry & Sensitive Skin, Sebum Reducing, Moisture Control & Minimizing Pores, Korean Skin Care, Gift for Women, Paraben-free, 4.23 Oz (120g)
6. SENGTERM Satin Bonnet Adjustable Sleep cap Double Layered Hair Bonnet Silky Cap for Women Curly Natural Long Hair (L, Black)
7. Cordless Water Flosse
...[truncated]...
```

Response:
```text
{
  "preferred_item": "maybe",
  "confidence": 0.5,
  "reason": "the answer is too ambiguous to be usable."
}
```

Parsed:
```json
{
  "preferred_item": "",
  "confidence": 0.5,
  "reason": "the answer is too ambiguous to be usable.",
  "parse_mode": "json",
  "parse_success": false,
  "ambiguous_preference": true
}
```

Repair note: Pairwise parser should handle explicit item ids, fenced A/B shorthand, and free-form preference text while preserving an explicit ambiguous failure case.
