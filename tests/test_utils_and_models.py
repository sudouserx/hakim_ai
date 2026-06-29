"""
Unit tests for utilities and foundation model adapters.

Covers:
  - RAGStore: keyword retrieval, category filtering, similar cases
  - Image utilities: tissue mask, stain normalizers, artifact detection
  - Foundation model adapters: UNI2Encoder, CONCHEncoder, PathChatVLM
"""
from __future__ import annotations

import math
import pytest

from hakim_ai.utils.rag_store import RAGStore, KnowledgeDocument, _tokenize, _tokenize_freq
from hakim_ai.utils.image_utils import (
    compute_tissue_mask,
    tissue_coverage,
    estimate_focus_quality,
    estimate_stain_quality,
    detect_artifacts,
    build_normalizer,
    extract_patch_coordinates,
)
from hakim_ai.foundation_models.uni_adapter import UNI2Encoder, _hash_to_vector
from hakim_ai.foundation_models.conch_adapter import CONCHEncoder, PathChatVLM


# ---------------------------------------------------------------------------
# RAGStore tests
# ---------------------------------------------------------------------------

class TestRAGStore:

    @pytest.fixture
    def store(self):
        return RAGStore()   # uses built-in mock knowledge base

    def test_retrieve_returns_documents(self, store):
        docs = store.retrieve("gastric cancer MSI", top_k=3)
        assert isinstance(docs, list)
        assert len(docs) <= 3

    def test_retrieve_returns_relevant_docs_first(self, store):
        docs = store.retrieve("MSI microsatellite instability TIL", top_k=5)
        # The MSI-specific document should rank highly
        contents = [d.content.lower() for d in docs]
        assert any("msi" in c or "microsatellite" in c or "til" in c for c in contents)

    def test_retrieve_top_k_respected(self, store):
        docs = store.retrieve("gastric cancer", top_k=2)
        assert len(docs) <= 2

    def test_retrieve_by_category_who_criteria(self, store):
        docs = store.retrieve_by_category("who_criteria", top_k=5)
        assert all(d.metadata.get("category") == "who_criteria" for d in docs)

    def test_retrieve_by_category_biomarker(self, store):
        docs = store.retrieve_by_category("biomarker", top_k=5)
        assert all(d.metadata.get("category") == "biomarker" for d in docs)

    def test_retrieve_by_category_guidelines(self, store):
        docs = store.retrieve_by_category("guidelines", top_k=3)
        assert all(d.metadata.get("category") == "guidelines" for d in docs)

    def test_get_similar_cases_returns_list(self, store):
        cases = store.get_similar_cases(top_k=3)
        assert isinstance(cases, list)
        assert len(cases) <= 3

    def test_similar_cases_have_required_keys(self, store):
        cases = store.get_similar_cases(top_k=2)
        for case in cases:
            assert "case_id" in case
            assert "diagnosis" in case
            assert "similarity_score" in case

    def test_empty_query_returns_documents(self, store):
        docs = store.retrieve("", top_k=3)
        assert isinstance(docs, list)

    def test_custom_documents(self):
        """RAGStore should work with custom documents."""
        custom_doc = KnowledgeDocument(
            doc_id="custom_001",
            content="Custom gastric cancer pathology note about HER2.",
            metadata={"category": "custom"},
        )
        store = RAGStore(documents=[custom_doc])
        docs = store.retrieve("HER2 gastric", top_k=1)
        assert len(docs) == 1
        assert docs[0].doc_id == "custom_001"

    def test_idf_computed_on_init(self, store):
        assert len(store.idf) > 0

    def test_tokenize_produces_lowercase(self):
        tokens = _tokenize("Gastric Cancer MSI-H")
        assert all(t.islower() for t in tokens)

    def test_tokenize_removes_punctuation(self):
        tokens = _tokenize("cancer, adenocarcinoma.")
        assert "," not in tokens
        assert "." not in tokens

    def test_term_freq_counts_correctly(self):
        freq = _tokenize_freq("cancer cancer gastric")
        assert freq["cancer"] == 2
        assert freq["gastric"] == 1


# ---------------------------------------------------------------------------
# Image utilities tests
# ---------------------------------------------------------------------------

class TestImageUtils:

    def test_compute_tissue_mask_returns_2d_list(self):
        thumbnail = [[[200, 180, 210]] * 16 for _ in range(16)]
        mask = compute_tissue_mask(thumbnail)
        assert isinstance(mask, list)
        assert len(mask) == 16
        assert all(isinstance(row, list) for row in mask)

    def test_tissue_mask_default_shape(self):
        mask = compute_tissue_mask(None)
        assert len(mask) == 16
        assert len(mask[0]) == 16

    def test_tissue_mask_values_are_bool(self):
        mask = compute_tissue_mask(None)
        for row in mask:
            for val in row:
                assert isinstance(val, bool)

    def test_tissue_coverage_returns_float(self):
        mask = compute_tissue_mask(None)
        cov = tissue_coverage(mask)
        assert isinstance(cov, float)
        assert 0.0 <= cov <= 1.0

    def test_tissue_coverage_empty_mask_returns_zero(self):
        cov = tissue_coverage([])
        assert cov == 0.0

    def test_tissue_coverage_all_tissue(self):
        mask = [[True] * 4 for _ in range(4)]
        assert tissue_coverage(mask) == 1.0

    def test_tissue_coverage_no_tissue(self):
        mask = [[False] * 4 for _ in range(4)]
        assert tissue_coverage(mask) == 0.0

    def test_focus_quality_returns_float_in_range(self):
        score = estimate_focus_quality(None)
        assert 0.0 <= score <= 1.0

    def test_stain_quality_returns_float_in_range(self):
        score = estimate_stain_quality(None)
        assert 0.0 <= score <= 1.0

    def test_detect_artifacts_returns_list(self):
        artifacts = detect_artifacts(None)
        assert isinstance(artifacts, list)

    def test_build_normalizer_macenko(self):
        norm = build_normalizer("macenko")
        assert norm is not None
        path = norm.normalize("/fake/slide.svs")
        assert "_normalized" in path or path.endswith(".svs")

    def test_build_normalizer_reinhard(self):
        norm = build_normalizer("reinhard")
        path = norm.normalize("/fake/slide.svs")
        assert "reinhard" in path

    def test_build_normalizer_passthrough(self):
        norm = build_normalizer("passthrough")
        path = norm.normalize("/fake/slide.svs")
        assert path == "/fake/slide.svs"

    def test_build_normalizer_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_normalizer("nonexistent_method")

    def test_extract_patch_coordinates_returns_list(self):
        mask = [[True] * 8 for _ in range(8)]
        coords = extract_patch_coordinates(mask, patch_size=512, thumbnail_downsample=64, top_k=5)
        assert isinstance(coords, list)
        assert len(coords) <= 5

    def test_extract_patch_coordinates_within_bounds(self):
        mask = [[True] * 4 for _ in range(4)]
        coords = extract_patch_coordinates(mask, patch_size=512, thumbnail_downsample=64, top_k=10)
        for x, y in coords:
            assert x >= 0 and y >= 0

    def test_extract_patch_coordinates_empty_mask(self):
        mask = [[False] * 4 for _ in range(4)]
        coords = extract_patch_coordinates(mask, patch_size=512, thumbnail_downsample=64, top_k=5)
        assert coords == []


# ---------------------------------------------------------------------------
# Foundation model adapter tests
# ---------------------------------------------------------------------------

class TestUNI2Encoder:

    @pytest.fixture
    def encoder(self):
        return UNI2Encoder(mock_mode=True)

    def test_embedding_dim_correct(self, encoder):
        assert encoder.embedding_dim == 1536

    def test_encode_patch_returns_vector(self, encoder):
        vec = encoder.encode_patch(None)
        assert isinstance(vec, list)
        assert len(vec) == 1536

    def test_encode_patch_values_in_range(self, encoder):
        vec = encoder.encode_patch(None)
        # Should be a unit vector, values in [-1, 1]
        for v in vec:
            assert -1.1 <= v <= 1.1

    def test_encode_patch_is_unit_vector(self, encoder):
        vec = encoder.encode_patch("mock_patch_data")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 0.001

    def test_encode_batch_returns_list_of_vectors(self, encoder):
        batch = [None, None, None]
        vecs = encoder.encode_batch(batch)
        assert len(vecs) == 3
        assert all(len(v) == 1536 for v in vecs)

    def test_hash_to_vector_deterministic(self):
        v1 = _hash_to_vector("test_seed", 128)
        v2 = _hash_to_vector("test_seed", 128)
        assert v1 == v2

    def test_hash_to_vector_different_seeds_differ(self):
        v1 = _hash_to_vector("seed_A", 64)
        v2 = _hash_to_vector("seed_B", 64)
        assert v1 != v2

    def test_real_mode_raises_when_no_weights(self):
        """Real mode should raise ImportError (missing torch) or NotImplementedError."""
        with pytest.raises((NotImplementedError, ImportError)):
            UNI2Encoder(mock_mode=False)


class TestCONCHEncoder:

    @pytest.fixture
    def encoder(self):
        return CONCHEncoder(mock_mode=True)

    def test_embedding_dim(self, encoder):
        assert encoder.embedding_dim == 512

    def test_encode_patch_correct_dim(self, encoder):
        vec = encoder.encode_patch(None)
        assert len(vec) == 512

    def test_encode_text_correct_dim(self, encoder):
        vec = encoder.encode_text("gastric cancer MSI-H")
        assert len(vec) == 512

    def test_encode_text_deterministic(self, encoder):
        v1 = encoder.encode_text("gastric cancer")
        v2 = encoder.encode_text("gastric cancer")
        assert v1 == v2

    def test_different_texts_produce_different_vectors(self, encoder):
        v1 = encoder.encode_text("MSI-H gastric cancer")
        v2 = encoder.encode_text("HER2 positive adenocarcinoma")
        assert v1 != v2


class TestPathChatVLM:

    @pytest.fixture
    def vlm(self):
        return PathChatVLM(mock_mode=True, seed=42)

    def test_describe_patch_returns_string(self, vlm):
        desc = vlm.describe_patch(None, prompt="Describe this H&E patch.")
        assert isinstance(desc, str)
        assert len(desc) > 20

    def test_description_cycles_through_templates(self, vlm):
        """Multiple calls should return different descriptions."""
        descriptions = [vlm.describe_patch(None) for _ in range(6)]
        # At least 2 should be different (6 calls across 6 templates)
        unique = set(descriptions)
        assert len(unique) >= 2

    def test_answer_question_msi_returns_msi_content(self, vlm):
        answer = vlm.answer_question(None, "Are there MSI features?")
        assert "MSI" in answer or "lymphocyte" in answer.lower() or "TIL" in answer

    def test_answer_question_lauren_returns_lauren_content(self, vlm):
        answer = vlm.answer_question(None, "What Lauren type is this?")
        assert "Lauren" in answer or "intestinal" in answer.lower() or "diffuse" in answer.lower()

    def test_answer_question_her2_returns_her2_content(self, vlm):
        answer = vlm.answer_question(None, "Any HER2 evidence?")
        assert "HER2" in answer

    def test_answer_question_generic_fallback(self, vlm):
        answer = vlm.answer_question(None, "What is the mitotic rate?")
        assert isinstance(answer, str)
        assert len(answer) > 10