"""
Marketplace trust & safety scoring.

Online marketplaces are attacked with synthetic identities: seller accounts
opened with GAN-generated profile photos, review farms whose accounts each
carry a different fake face, and identity-verification selfies that were
never taken by a real person. This module applies the face classifier to
exactly those surfaces and turns per-image verdicts into one listing-level
risk score plus a moderation decision.

Scope, stated honestly: the classifier is trained on faces. Images with no
detectable face are reported as OUT_OF_SCOPE and excluded from the score
rather than given a number the model cannot justify. Product photography is
deliberately not scored - an AI-rendered product shot is both common and
usually legitimate, and it is outside what this model learned. See the
README limitations section.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from model import DeepfakeClassifier
from video import FACE_CASCADE_PATH


class ImageRole(str, Enum):
    """Where in the marketplace an image appears. Drives how much it counts."""

    SELLER_AVATAR = "seller_avatar"
    VERIFICATION_SELFIE = "verification_selfie"
    REVIEWER_AVATAR = "reviewer_avatar"


# A synthetic seller avatar or verification selfie is a direct identity-fraud
# signal. A single synthetic reviewer avatar is weaker on its own - one review
# proves little - so it carries less weight in the aggregate.
ROLE_WEIGHTS: dict[ImageRole, float] = {
    ImageRole.SELLER_AVATAR: 1.0,
    ImageRole.VERIFICATION_SELFIE: 1.0,
    ImageRole.REVIEWER_AVATAR: 0.6,
}

# Roles that identify the account holder. A confident synthetic verdict on one
# of these escalates the listing on its own, even when the average stays low.
IDENTITY_ROLES = (ImageRole.SELLER_AVATAR, ImageRole.VERIFICATION_SELFIE)

BLOCK_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.40
IDENTITY_ESCALATION_THRESHOLD = 0.70


class Decision(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    BLOCK = "block"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class ImageAssessment:
    role: ImageRole
    filename: str
    in_scope: bool
    prob_fake: Optional[float] = None
    label: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "filename": self.filename,
            "in_scope": self.in_scope,
            "prob_fake": self.prob_fake,
            "label": self.label,
            "note": self.note,
        }


class MarketplaceScorer:
    def __init__(self, classifier: DeepfakeClassifier):
        self.classifier = classifier
        self.face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    def _has_face(self, image: Image.Image) -> bool:
        rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return len(faces) > 0

    def assess_image(self, image: Image.Image, role: ImageRole, filename: str) -> ImageAssessment:
        if not self._has_face(image):
            return ImageAssessment(
                role=role,
                filename=filename,
                in_scope=False,
                note="No face detected - outside the classifier's trained domain, not scored.",
            )

        prediction = self.classifier.predict(image)
        return ImageAssessment(
            role=role,
            filename=filename,
            in_scope=True,
            prob_fake=prediction["prob_fake"],
            label=prediction["label"],
        )

    def score_listing(self, assessments: list[ImageAssessment]) -> dict:
        scored = [a for a in assessments if a.in_scope]

        if not scored:
            return {
                "decision": Decision.INSUFFICIENT_EVIDENCE.value,
                "risk_score": None,
                "reasons": ["No image contained a detectable face, so no verdict can be justified."],
                "images": [a.to_dict() for a in assessments],
            }

        weights = [ROLE_WEIGHTS[a.role] for a in scored]
        risk_score = float(
            sum(a.prob_fake * w for a, w in zip(scored, weights)) / sum(weights)
        )

        reasons: list[str] = []
        flagged_identity = [
            a for a in scored
            if a.role in IDENTITY_ROLES and a.prob_fake >= IDENTITY_ESCALATION_THRESHOLD
        ]
        for a in flagged_identity:
            reasons.append(
                f"{a.role.value} '{a.filename}' scored {a.prob_fake:.0%} synthetic - "
                "identity images are weighted as direct fraud signals."
            )

        flagged_reviewers = [
            a for a in scored
            if a.role is ImageRole.REVIEWER_AVATAR and a.prob_fake >= 0.5
        ]
        if len(flagged_reviewers) >= 2:
            reasons.append(
                f"{len(flagged_reviewers)} reviewer avatars scored synthetic - "
                "a pattern consistent with a review farm."
            )

        if risk_score >= BLOCK_THRESHOLD:
            decision = Decision.BLOCK
        elif risk_score >= REVIEW_THRESHOLD or flagged_identity or len(flagged_reviewers) >= 2:
            decision = Decision.REVIEW
        else:
            decision = Decision.APPROVE

        if not reasons:
            reasons.append(f"Weighted risk score {risk_score:.0%} across {len(scored)} scored image(s).")

        skipped = len(assessments) - len(scored)
        if skipped:
            reasons.append(f"{skipped} image(s) excluded: no detectable face.")

        return {
            "decision": decision.value,
            "risk_score": risk_score,
            "reasons": reasons,
            "images": [a.to_dict() for a in assessments],
        }
