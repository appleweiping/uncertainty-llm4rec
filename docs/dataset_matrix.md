# Dataset Matrix

MovieLens 1M is only a local real-data sanity check. After the MovieLens
pipeline works, Storyflow / TRUCE-Rec must move to at least one Amazon Reviews
2023 category for full-data observation. The project must not remain in small
sanity mode.

| dataset | domain | role in Storyflow | title field quality | interaction type | timestamp availability | metadata availability | expected scale | local feasible or server-only | preprocessing setting | split setting | current status | risks | next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Synthetic fixtures | Test-only synthetic records | Unit tests and pipeline sanity only; never paper results | Controlled | Synthetic interactions | Controlled | Minimal | Tiny | Local | Hand-authored fixtures | Test-specific | Implemented for tests | Misuse as results | Keep labeled synthetic and limited to tests |
| MovieLens 1M | Movies | Local real-data sanity check for catalog, splits, grounding, and mock observation | Good human-readable movie titles | Ratings as implicit interactions after preprocessing | Yes | Genres | About 1M raw ratings | Local feasible | User filtering, k-core, title cleaning, popularity buckets | leave-last-one/two and global chronological supported | Download fallback and sanity prepare verified | Movie domain is small and not enough for final claims | Use for Phase 2A mock and low-cost API pilot only |
| Amazon Reviews 2023 - Beauty | E-commerce | First full e-commerce category candidate | Product titles can be noisy but realistic | Reviews/ratings as implicit interactions | Yes | Product metadata where available | Full category scale, server-oriented | Server/full run | Category config, k-core, item/user filtering, metadata join | Rolling examples plus chronological global split; leave-last variants where useful | Config and HF entry planned | HF access, storage, preprocessing runtime | Build full-data downloader/processor and run on server or suitable local disk |
| Amazon Reviews 2023 - Sports_and_Outdoors | E-commerce | Full e-commerce category for robustness | Product titles with brands/models | Reviews/ratings as implicit interactions | Yes | Product metadata where available | Full category scale, server-oriented | Server/full run | Category config, k-core, item/user filtering, metadata join | Rolling examples plus chronological global split | Config and HF entry planned | Scale and title normalization complexity | Prepare after Beauty pipeline is stable |
| Amazon Reviews 2023 - Video_Games | Games/e-commerce | Title-rich game-like category for popular-title confidence and tail underconfidence | Strong title signal, often franchise-heavy | Reviews/ratings as implicit interactions | Yes | Product/game metadata where available | Full category scale, server-oriented | Server/full run | Category config, k-core, item/user filtering, metadata join | Rolling examples plus chronological global split | Config and HF entry planned | Franchise/title ambiguity; popularity prior can dominate | Prioritize for confidence-popularity observation after Beauty |
| Amazon Reviews 2023 - Books | Books | Long-tail title-rich domain, likely strong for server full run | Very rich long-tail titles | Reviews/ratings as implicit interactions | Yes | Product metadata where available | Large, server-oriented | Server-only for full run | Category config needed, k-core, title/author metadata join | Rolling examples plus chronological global split | Not yet configured in repo | Very large scale; duplicate titles/editions | Add dataset config and storage/run plan |
| Steam / Games | Games | Cross-domain title-generation validation | Good game titles, source-dependent | Play/review interactions depending on source | Source-dependent | Source-dependent | Medium to large | Likely server/full for robust run | Planned after source selection | Chronological split if timestamps available | Planned placeholder config | Source/license must be confirmed | Verify source/license and update config |
| Yelp / POI optional | POI/local business | Optional cross-domain validation, not a current blocker | Business names can be ambiguous | Reviews/check-ins | Yes in common releases | Business/category/location metadata | Large | Server-oriented if used | Future config only | Chronological split if selected | Optional | License/access and title ambiguity | Defer until Amazon and games flows are stable |

## Current Priority

1. Keep MovieLens 1M as local sanity and mock/API-pilot substrate.
2. Build full-data preparation for Amazon Reviews 2023 Beauty.
3. Extend to Video_Games or Sports_and_Outdoors once the first Amazon category
   is reproducible.
4. Add Books, Steam, or Yelp only when they serve a specific research question
   and the source/license path is clear.
