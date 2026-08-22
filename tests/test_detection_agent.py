import numpy as np
from PIL import Image

from scripts.analyze_agent_headroom import canonical_image_key
from src.detection_agent.evidence_builder import build_evidence_board, patch_concentration, top_regions
from src.detection_agent.fusion import ConservativeFusion, calibrate_patchcore_score
from src.detection_agent.conflict_verifier import ConflictVerifier, VerificationGate
from src.detection_agent.pipeline import DetectionAgent
from src.detection_agent.router import BudgetRouter
from src.detection_agent.schemas import ExpertEvidence, ReviewEvidence, VerificationEvidence


def _expert(tmp_path, probability=0.52):
    query = tmp_path / "query.png"
    reference = tmp_path / "reference.png"
    Image.new("RGB", (128, 96), "gray").save(query)
    Image.new("RGB", (128, 96), "white").save(reference)
    scores = np.zeros((14, 14), dtype=float)
    scores[3, 9] = 1.0
    regions = top_regions(scores, (128, 96), top_k=2)
    return ExpertEvidence(
        image_path=str(query), category="bottle", score=0.051, threshold=0.05,
        probability=probability, decision="NG", patch_scores=scores,
        reference_path=str(reference), regions=regions,
        concentration=patch_concentration(scores),
    )


def test_canonical_paths_match_hosts():
    linux = "/mnt/host_a/datasets/mvtec/bottle/test/good/000.png"
    server = "/mnt/host_b/datasets/mvtec/bottle/test/good/000.png"
    assert canonical_image_key(linux) == canonical_image_key(server)


def test_calibration_keeps_threshold_at_half():
    assert calibrate_patchcore_score(0.05, 0.05) == 0.5
    assert calibrate_patchcore_score(0.06, 0.05) > 0.5


def test_budget_router_selects_exact_budget(tmp_path):
    items = [_expert(tmp_path, probability=value) for value in (0.1, 0.45, 0.51, 0.8)]
    selected = BudgetRouter(review_budget=0.5, concentration_weight=0.0).select(items)
    assert selected == {1, 2}


def test_region_guard_falls_back_to_expert(tmp_path):
    expert = _expert(tmp_path, probability=0.52)
    review = ReviewEvidence("NG", 0.95, "crack", "elsewhere", False, True, 10.0)
    result = DetectionAgent(ConservativeFusion()).decide(expert, reviewed=True, review=review)
    assert result.final_score == expert.probability
    assert result.fallback_reason == "region_disagreement"


def test_high_confidence_aligned_ng_raises_score(tmp_path):
    expert = _expert(tmp_path, probability=0.52)
    review = ReviewEvidence("NG", 0.95, "crack", "aligned", True, True, 10.0)
    result = DetectionAgent(ConservativeFusion()).decide(expert, reviewed=True, review=review)
    assert result.final_score > expert.probability
    assert result.review_applied


def test_evidence_board_shape(tmp_path):
    expert = _expert(tmp_path)
    board = build_evidence_board(expert, panel_size=128)
    assert board.image.size == (256, 312)
    assert len(board.regions) == 2


def test_verification_gate_accepts_only_consistent_override(tmp_path):
    expert = _expert(tmp_path, probability=0.48)
    expert.decision = "OK"
    review = ReviewEvidence("NG", 0.95, "crack", "aligned", True, True, 10.0)
    verification = VerificationEvidence(True, "NG", 0.96, True, "visible crack", True, 12.0)
    accepted, reason = VerificationGate().accept(expert, review, verification)
    assert accepted
    assert reason == "override_confirmed"


def test_verification_gate_rejects_ambiguous_override(tmp_path):
    expert = _expert(tmp_path, probability=0.48)
    expert.decision = "OK"
    review = ReviewEvidence("NG", 0.95, "crack", "aligned", True, True, 10.0)
    verification = VerificationEvidence(False, "OK", 0.97, False, "normal texture", True, 12.0)
    accepted, reason = VerificationGate().accept(expert, review, verification)
    assert not accepted
    assert reason == "override_rejected"


def test_verification_gate_advisory_mode_never_overrides(tmp_path):
    expert = _expert(tmp_path, probability=0.48)
    expert.decision = "OK"
    review = ReviewEvidence("NG", 0.99, "crack", "aligned", True, True, 10.0)
    verification = VerificationEvidence(True, "NG", 0.99, True, "visible crack", True, 12.0)
    accepted, reason = VerificationGate(allow_overrides=False).accept(
        expert, review, verification
    )
    assert not accepted
    assert reason == "advisory_only"


class _FakeResult:
    decision = "NG"
    confidence = 0.93
    reason = "localized crack"
    parse_ok = True
    latency_ms = 7.0
    peak_mem_mb = 123.0
    raw = '{"confirm_override":true,"decision":"NG","confidence":0.93,"region_agreement":true,"reason":"localized crack"}'


class _FakeClient:
    prompt = ""

    def infer(self, _board):
        return _FakeResult()


def test_conflict_verifier_parses_explicit_confirmation(tmp_path):
    expert = _expert(tmp_path, probability=0.48)
    expert.decision = "OK"
    review = ReviewEvidence("NG", 0.95, "crack", "aligned", True, True, 10.0)
    client = _FakeClient()
    result = ConflictVerifier(client).verify(expert.image_path, expert, review)
    assert result.parse_ok
    assert result.confirm_override
    assert result.region_agreement
    assert "category: bottle" in client.prompt
