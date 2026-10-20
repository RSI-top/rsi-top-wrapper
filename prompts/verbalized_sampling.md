## Verbalized proposal distribution

Generate the directions as one joint portfolio sampled across the distribution
of credible interventions, not as variations of the single most typical answer.
Include credible lower-probability modes as well as common modes.
{mixture}

Add a numeric `verbalized_probability` between 0 and 1 to every direction. It
estimates how likely an ordinary planner would be to independently propose that
direction. It is generation typicality, not probability of success, and the
values need not sum to 1. Do not weaken a direction merely to give it a lower
probability. Directions are substantively distinct only when they test
materially different bottleneck -> mechanism -> expected-effect hypotheses;
changing only a target, parameter, or wording is insufficient.
