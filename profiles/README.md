# Mechanism profiles

`profile.example.json` is a starting body for `publish_profile.py`. It is not published
and not read by anything at runtime: validators only ever use what the API serves.

`version`, `publish_block` and `activation_block` are filled in by the publish command.

## Grader rotation

The three models are the strong, cheap end of the 12-model benchmark. The model the
subnet grades with today is deliberately absent: it finds roughly an eighth of the
claims the field finds between them, and because the grader's own claim set is the
recall denominator, a thin grader does not save money so much as shrink what a
submission is measured against.

Weights are a dial. Raising the share of the cheapest model lowers cost per article and
lowers the claims found with it, so the two move together and should be argued together.

The benchmark behind these weights covered 20 articles. Rerun it at a larger sample
before treating the ordering as settled.
