# Project Report — Synthetic Identity Detection for Marketplace Trust & Safety

**Subject:** Technologie handlu elektronicznego (e-commerce)
**Author:** Valentyn Hotsulenko
**Student ID:** 494098
**Programme:** Technologie Komputerowe
**Repository:** https://github.com/Quertikk/AI-detect

---

## 1. The e-commerce problem

Every open marketplace faces the same structural weakness: **anyone can open a seller account, and the platform has almost no cheap way to verify that a real person is behind it.** Generative face models removed the last practical barrier to abusing this.

Three concrete attack patterns motivate this work:

**Synthetic seller identities.** A fraudulent seller needs a profile photo that survives a glance. Stolen photos can be caught by reverse image search; a freshly generated face cannot, because it has never existed anywhere before. This is the decisive advantage generated faces give the attacker.

**Review farms.** Marketplace ranking is driven by review volume and score. A farm operating hundreds of accounts needs hundreds of distinct, plausible profile photos. Generated faces supply them at zero marginal cost. The manipulated ranking then misdirects real buyers.

**Identity-verification bypass.** Platforms that ask for a selfie during KYC or payout verification are, in effect, asking for an image — and an image can be generated.

The financial consequence is not abstract. Fraudulent sellers cause chargebacks, refunds, and support cost; manipulated reviews degrade the ranking signal the marketplace's whole value proposition rests on.

## 2. What this project contributes

A moderation service that scores the images of a marketplace listing which actually contain faces, and converts those scores into a **moderation decision with a stated justification**: `approve`, `review`, `block`, or `insufficient_evidence`.

The core detector is documented in the companion report (`report-ai-models.md`); this report covers the e-commerce application layer built on top of it: `backend/ecommerce.py` and the `POST /api/moderate/listing` endpoint.

### 2.1 The scoping decision that shapes the whole design

The classifier is trained on **faces**. That single fact determines what this system may and may not claim.

A naive product would run every image in a listing — including product photography — through the model and print a number. That number would be meaningless: a photo of a running shoe is far outside the distribution the model was trained on, and the model has no basis for a verdict on it.

This system therefore **detects whether a face is present before scoring anything**. Images with no detectable face are returned as `in_scope: false` with an explicit note and are excluded from the aggregate. If no image in a listing contains a face, the verdict is `insufficient_evidence` — not a fabricated score.

This is a deliberate design choice, and it is arguably the most important one in the project: **a trust & safety system that produces confident-looking numbers it cannot justify is worse than one that admits the limits of its evidence,** because moderators learn to trust the numbers.

AI-generated *product* imagery is also, notably, both common and usually legitimate — retailers routinely use generated or heavily edited product renders. Flagging it as fraud would generate false positives against honest sellers.

## 3. Design

### 3.1 Image roles and weighting

Not every face in a listing carries the same evidential weight, so each image is submitted with a role:

| Role | Weight | Rationale |
|---|---|---|
| `seller_avatar` | 1.0 | Identifies the account holder — direct identity-fraud signal |
| `verification_selfie` | 1.0 | Submitted as proof of a real person; synthetic here is decisive |
| `reviewer_avatar` | 0.6 | One fake reviewer proves little; the *pattern* is what matters |

### 3.2 Decision policy

The listing risk score is the role-weighted mean of P(synthetic) across in-scope images. Two escalation rules sit on top of the raw average, because averaging alone hides exactly the cases that matter most:

**Identity escalation.** If a `seller_avatar` or `verification_selfie` scores ≥ 0.70 synthetic, the listing escalates to at least `review` regardless of the average. Without this rule, one fraudulent seller photo could be diluted below threshold by a handful of genuine reviewer avatars — which is precisely what an attacker would arrange.

**Review-farm pattern.** Two or more reviewer avatars scoring ≥ 0.50 escalates to `review`. A single synthetic reviewer avatar is weak evidence; a cluster is a pattern.

Final thresholds: `block` at ≥ 0.75, `review` at ≥ 0.40 or on either escalation rule, otherwise `approve`.

**Nothing is auto-blocked into a human vacuum.** Every response carries a `reasons` array in plain language, because a moderator has to be able to justify an account action to the seller, and increasingly to a regulator.

### 3.3 API

```
POST /api/moderate/listing
  files: image[]        one or more images
  roles: string[]       one role per image, positionally matched
```

Returns the decision, the risk score, the human-readable reasons, and the per-image breakdown including which images were excluded and why.

## 4. Evaluation

The policy was exercised against four fraud scenarios plus two scoping cases, using held-out test images the detector never saw during training.

| Scenario | Composition | Decision | Risk |
|---|---|---|---|
| Legitimate seller | real avatar + 2 real reviewers | `approve` | 0.02 |
| Synthetic seller identity | **fake** avatar + real reviewer | `review` | 0.64 |
| Review farm | real avatar + 3 **fake** reviewers | `review` | 0.58 |
| Fully synthetic account | fake avatar + fake selfie + fake reviewer | `block` | 0.88 |
| Product photo only (no face) | 1 no-face image | `insufficient_evidence` | — |
| Mixed | real avatar + no-face product photo | `approve` | 0.00 (1 excluded) |

Both escalation rules fired as intended:

- The synthetic-seller case reached `review` on the identity rule, with the reason *"seller_avatar scored 100% synthetic — identity images are weighted as direct fraud signals."*
- The review-farm case reached `review` on the pattern rule, with *"3 reviewer avatars scored synthetic — a pattern consistent with a review farm."*

Input validation was verified: mismatched file/role counts and unknown role names both return HTTP 400 with an actionable message.

## 5. Integration in a real marketplace

Realistically this sits at two points in the platform lifecycle:

1. **Account onboarding** — score the avatar and any verification selfie before the seller can list. This is where the cost of a false negative is highest and the cost of a false positive is lowest, since the seller is already in a registration flow and can be asked for another photo.
2. **Review ingestion** — score reviewer avatars in batch, looking for clusters rather than individuals.

Operational cautions that a deployment would have to respect:

- **The threshold is a business decision, not a technical one.** The detector's ROC-AUC of 0.948 means the ranking is good; where to cut it depends on the relative cost of blocking an honest seller versus admitting a fraudster. That is a policy call for the platform.
- **`block` should gate a human queue, not an irreversible ban.** An 88%-accurate model will wrongly flag honest sellers, and wrongly banning a legitimate merchant is a serious commercial and legal harm.
- **Adversarial pressure is guaranteed.** Fraudsters iterate. A detector trained on one generator family will decay as generators change — this needs retraining as a standing operational cost, not a one-off.

## 6. Limitations

1. **One generator family in training** (StyleGAN). Diffusion-generated avatars are untested and likely weaker — the most important gap for real deployment.
2. **Face detection uses a Haar cascade**, which misses off-angle, occluded, and heavily stylised avatars. An avatar that is a cartoon or a logo is correctly reported out of scope, but a real face at an extreme angle may be too.
3. **Product imagery is not assessed at all** — by design, as argued in §2.1.
4. **No cross-account linking.** Real review-farm detection also uses account-creation timing, IP and device fingerprints, and writing-style similarity. This system contributes one signal and should be fused with those, not used alone.
5. **In-memory service, single process.** Suitable for demonstration, not production throughput.

## 7. Conclusions

Synthetic identity is a live and growing e-commerce fraud vector, and it is one where an image classifier can contribute a genuinely useful signal — provided it is applied only where it is competent.

The engineering contribution here is less the model than the **discipline around it**: roles that weight evidence by what it actually proves, escalation rules that catch the dilution attack a plain average would miss, refusal to score out-of-domain images, and a reason string attached to every decision so a human can act on it and defend it.

Next steps: add diffusion-generated avatars to training; replace the Haar cascade with a modern detector; fuse these scores with behavioural signals; and calibrate thresholds against measured chargeback cost rather than a default 0.5.

## 8. Running the moderation endpoint

```bash
cd backend && uvicorn app:app
```

```bash
curl -X POST http://localhost:8000/api/moderate/listing \
  -F "files=@avatar.jpg"   -F "roles=seller_avatar" \
  -F "files=@reviewer.jpg" -F "roles=reviewer_avatar"
```

Source: `backend/ecommerce.py` (roles, weighting, policy), `backend/app.py` (endpoint and validation).
