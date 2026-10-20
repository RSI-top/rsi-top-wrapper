You are removing only obvious semantic duplicates from a frozen pool of
intervention directions.

Remove a direction only when a kept direction proposes substantially the same
mechanism, concrete action, and expected causal effect. Sharing a broad topic,
bottleneck, target, or direction family is not enough. Keep variants whose
implementation, scope, composition, target state, or falsifiable effect differs
materially. When uncertain, keep both.

Do not judge whether a unique direction is promising, exhausted, well evidenced,
or likely to succeed. Within an obvious duplicate group, keep the clearest and
most executable representative. The number kept is variable.

Return exactly one JSON object:

{
  "kept_direction_ids": ["supplied direction id", "..."],
  "removed_directions": [
    {
      "direction_id": "removed supplied direction id",
      "duplicate_of": "kept supplied direction id",
      "reason": "brief concrete reason the two directions are obvious semantic duplicates"
    }
  ]
}

Every supplied direction ID must appear exactly once: either in
`kept_direction_ids` or as one `removed_directions[].direction_id`.
Every `duplicate_of` must name the kept obvious duplicate. Copy every ID
byte-for-byte exactly as supplied; for example, never abbreviate
`/direction_00` as `/d00`. Do not invent IDs.

Candidate directions:

{{candidate_directions_json}}
