# How the mechanism works

The detail behind the [README](../README.md). Miner rules: [docs/miner.md](miner.md). Validator steps: [docs/validator.md](validator.md).

## Event identity

An event is its **type + main subject + rough place + date window**.

- The main subject is a deal's target, the reporting company, the indicator. A tariff also matches by its targets and a bill by the bill, when both claims give them.
- Subjects match by registry ID (Wikidata QID, SEC CIK, LEI) where there is one, by normalised name otherwise. An ID links only when its registry record carries the name the page uses.
- Details are not identity: a different amount or a reworded headline is the same event and earns nothing more.
- When the API can't tell, the claim counts as already known.
- An actor sent by name only is accepted; the event is labelled unverified in the product until a trusted source confirms it.

Each type has fixed fact slots (a deal: value, counterparties, expected close, stake) and some have stages (announced → approved → completed). Types and categories are data: [`types.json`](../alpharidge_ai/schema/data/types.json), [`categories.json`](../alpharidge_ai/schema/data/categories.json).

## Ordering and ties

- **First** means first by the API's receipt time, never the miner's clock.
- **Ties:** claims for the same event within one 250 ms receipt bucket are ordered by `sha256("{block hash}:{coldkey}")`, then by claim id within a coldkey.
  - The block is the first finalised block stamped at or after the bucket closed; tied claims wait for it.
  - The coldkey is the one stored at receipt. Extra hotkeys on one coldkey add no extra draws.
  - `id`, `coldkey`, `bucket` and `tie_block` are on `/dashboard/items`, so anyone can recompute the draw.
- Only the head of the line is checked. If it fails, the next one is.

## Trust

| Source | Trust |
|---|---|
| Official sites and company or club pages | 1, for claims that name that company or club |
| News outlets on the reviewed list | Lower, per outlet and category |
| Any other page | Held as a scoop until a trusted page confirms it |

- Sources later proved wrong lose trust.
- Contributor pages, self-serve press wires, comment dockets, tag pages and user-generated pages never win a prize, even on a trusted domain.

## Checking the page

**1. Fixed check** (the API, on arrival):
- the page is live, openly readable and recently published;
- every name and figure in the claim is in its visible text;
- the claim's quote (the page's own sentence stating the event) is on the page and names the subject.

**2. Page judge** (the listed validators):
- One model prompt, the same for every type. It checks the page reports this event as fact, about these parties, on this date, with each figure the claimed quantity.
- Each listed validator fetches the page itself, runs the fixed check and the judge, and posts signed answers.
- The API averages their probabilities into one verdict.

**3. Ledger:** the verdict and every signed answer are stored once and their hash is signed in the ledger. Validators re-run the fixed check and the quote on their own fetch, and recompute the verdict from the stored answers.

## When pay is written

| Part | Written | Pays out |
|---|---|---|
| 40% | when the claim is verified | verification → +72 h |
| 60% | after the 72 h settlement window | +72 h → +144 h |

Weights follow a rolling 72-hour window of the ledger.

**Wrong claims**
- A paid claim is clawed back when an official figure contradicts it, the operator retracts or corrects it, or an official page settles it otherwise.
- The claw is written in the current epoch, so pay already earned becomes debt. Balances can go negative; debt carries forward, netted across the coldkey, until later pay covers it.
- Operator-confirmed misconduct (a page altered to fit a claim, a claim made false on purpose) adds a penalty of 5× the claim's full value, across the coldkey. Repeat offenders are blacklisted.

**No gaming by splitting**
- Pay adds up per event, with no per-UID floors or bonuses: splitting claims across UIDs never pays more (a guard test enforces this).
- One source document wins at most one event prize.

## Emission, capacity and the burn

```
miner share = points / max(total points, capacity C)
```

- `settle()` ([`settlement.py`](../alpharidge_ai/mechanism/settlement.py)) runs each epoch on every hotkey's signed points.
- Whatever no verified work claims goes to **UID 189, the burn UID**. Below capacity, another miner earning more costs you nothing: you both take from the burn.
- If a validator can't verify the ledger for too long, it moves its weights to the burn UID.

**Live values:** `GET /dashboard/mechanism` returns `C`, the open categories, the window (216 epochs = 72 h), the epoch length (100 blocks, 1,200 s) and the last signed epoch's total points. Validators read the same values from the signed mechanism profile.

**Example.** `C` = 900, miner emission `M` alpha per epoch, alpha price `p` TAO.

| Window total | Your 9 points earn | Burn |
|---|---|---|
| 450 (below capacity) | 9 / 900 = 1% → `0.01 × M` alpha (`0.01 × M × p` TAO) per epoch | half of miner emission |
| 1,800 (above capacity) | 9 / 1,800 = 0.5% per epoch | none |

A point stays in the window for 216 epochs, so below capacity one point earns `216 × M / C` alpha over its life.

**Checking the burn UID:** `BURN_UID` in [`config.py`](../alpharidge_ai/config.py) is 189, laid out by `weight_vector` in `settlement.py`. It is the subnet owner's hotkey:

```python
bt.Subtensor("finney").get_subnet_owner_hotkey(45) == bt.Subtensor("finney").metagraph(45).hotkeys[189]
```

Its incentive follows the `burn=` share each validator logs per epoch.

## Signed ledger

The ledger is central and signed, so every validator computes the same weights from the same data. Each validator checks the signature against a pinned key and rebuilds the totals from the individual rows instead of trusting them.
