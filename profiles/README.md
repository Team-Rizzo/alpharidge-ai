# Mechanism profiles

`profile.example.json` shows the shape of a profile body. Its values are placeholders; the
operating values are in the signed profile the API serves to validators.

`version`, `publish_block` and `activation_block` are set when a profile is published.

Nothing here is read at runtime. Validators use only the profile the API serves, after
checking its signature against the pinned ledger key.
