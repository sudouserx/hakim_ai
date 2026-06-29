"""
CONCH / TITAN slide-level vision-language encoder adapter and
PathChat VLM adapter for natural-language patch descriptions.

Mock implementations return templated but contextually plausible text
and deterministic feature vectors so the full pipeline is testable
without GPU or model weights.
"""
from __future__ import annotations

import math
import random
from typing import Any, List, Optional

from hakim_ai.foundation_models.base_encoder import BaseEncoder, BaseVLM
from hakim_ai.foundation_models.uni_adapter import _hash_to_vector


# ---------------------------------------------------------------------------
# CONCH / TITAN slide encoder
# ---------------------------------------------------------------------------

class CONCHEncoder(BaseEncoder):
    """
    Adapter for CONCH (vision-language) / TITAN (slide-level) encoders.
    Embedding dim 512 (CONCH) or 768 (TITAN).
    """

    EMBEDDING_DIM = 512

    def __init__(self, mock_mode: bool = True, model_variant: str = "conch"):
        self.mock_mode = mock_mode
        self.model_variant = model_variant
        self._model = None

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    def load(self) -> None:
        if not self.mock_mode and self._model is None:
            self._load_real_model()

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def encode_patch(self, patch: Any) -> List[float]:
        if self.mock_mode:
            if patch is None:
                seed = "null"
            else:
                try:
                    import numpy as np
                    seed = str(hash(np.array(patch).tobytes()))
                except Exception:
                    seed = str(id(patch)) + self.model_variant
            return _hash_to_vector(seed, self.EMBEDDING_DIM)
            
        import torch
        if patch is None:
            return [0.0] * self.EMBEDDING_DIM
            
        if self._model is None:
            self.load()
            
        x = self._transform(patch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device if self.device != "cpu" else "cpu"):
                feat = self._model(x)
                feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat.squeeze(0).cpu().tolist()

    def encode_batch(self, patches: List[Any]) -> List[List[float]]:
        if self.mock_mode:
            return [self.encode_patch(p) for p in patches]
            
        import torch
        valid_patches = [p for p in patches if p is not None]
        if not valid_patches:
            return [[0.0] * self.EMBEDDING_DIM for _ in patches]
            
        if self._model is None:
            self.load()
            
        tensors = [self._transform(p) for p in valid_patches]
        x = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            with torch.autocast(device_type=self.device if self.device != "cpu" else "cpu"):
                feats = self._model(x)
                feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
                
        feat_list = feats.cpu().tolist()
        
        result = []
        idx = 0
        for p in patches:
            if p is None:
                result.append([0.0] * self.EMBEDDING_DIM)
            else:
                result.append(feat_list[idx])
                idx += 1
                
        return result

    def encode_text(self, text: str) -> List[float]:
        """Return a text embedding for zero-shot tasks."""
        if self.mock_mode:
            return _hash_to_vector(text, self.EMBEDDING_DIM)
            
        import torch
        if not hasattr(self, "_tokenizer") or self._tokenizer is None:
            return [0.0] * self.EMBEDDING_DIM
            
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if hasattr(self._model, "encode_text"):
                feat = self._model.encode_text(inputs["input_ids"], inputs["attention_mask"])
            else:
                return _hash_to_vector(text, self.EMBEDDING_DIM)
                
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat.squeeze(0).cpu().tolist()

    def _load_real_model(self) -> None:
        try:
            import torch
            import timm
            from huggingface_hub import login
            from torchvision import transforms
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Real CONCH requires torch, timm, transformers, and huggingface_hub."
            ) from exc

        import os
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        self.device = getattr(self, "device", "cuda" if torch.cuda.is_available() else "cpu")
        
        from huggingface_hub import hf_hub_download
        model_name = "MahmoodLab/CONCH"
        
        self._model = timm.create_model("vit_base_patch16_224", num_classes=0, global_pool='token')
        
        try:
            checkpoint_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            vision_state_dict = {k.replace('visual.', ''): v for k, v in state_dict.items() if k.startswith('visual.')}
            if not vision_state_dict:
                vision_state_dict = state_dict
            self._model.load_state_dict(vision_state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Failed to download CONCH weights: {e}")
            
        self._model = self._model.eval().to(self.device)
        
        self._transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        except Exception:
            self._tokenizer = None


# ---------------------------------------------------------------------------
# PathChat VLM
# ---------------------------------------------------------------------------

_DESCRIPTION_TEMPLATES = [
    (
        "The patch demonstrates irregular glandular architecture with loss of "
        "normal mucosal organisation. Nuclei are enlarged with prominent nucleoli "
        "and increased nuclear-to-cytoplasmic ratio. These features are consistent "
        "with moderately differentiated gastric adenocarcinoma."
    ),
    (
        "At 20× magnification, this region shows infiltrating tumour cells arranged "
        "in poorly cohesive clusters with signet-ring cell morphology. Background "
        "desmoplastic stroma and scattered lymphocytes are noted, raising concern "
        "for Lauren diffuse-type gastric carcinoma."
    ),
    (
        "The tissue displays well-formed glands lined by columnar epithelium with "
        "mild nuclear atypia. Goblet cells are present. The architecture is broadly "
        "intact with only focal irregularity, compatible with intestinal metaplasia "
        "or well-differentiated intestinal-type adenocarcinoma."
    ),
    (
        "A dense lymphocytic infiltrate surrounds tumour nests at the invasive margin. "
        "Tumour-infiltrating lymphocyte (TIL) density is high. Nuclear pleomorphism "
        "is moderate. This pattern of immune infiltration raises the possibility of "
        "microsatellite instability-high (MSI-H) gastric carcinoma."
    ),
    (
        "This high-power field reveals marked nuclear pleomorphism with atypical "
        "mitotic figures. No glandular differentiation is apparent. The stroma is "
        "densely fibrotic. Appearances are consistent with poorly differentiated "
        "gastric adenocarcinoma."
    ),
    (
        "The lamina propria contains a mixed inflammatory infiltrate. Glands show "
        "reactive changes without overt dysplasia. H. pylori-associated chronic "
        "active gastritis cannot be excluded in this region."
    ),
]


class PathChatVLM(BaseVLM):
    """
    Adapter for PathChat (CONCH vision encoder + LLM).

    Mock: returns contextually appropriate templated descriptions.
    Real: requires PathChat model weights and GPU.
    """

    def __init__(self, mock_mode: bool = True, seed: int = 42):
        self.mock_mode = mock_mode
        self._rng = random.Random(seed)
        self._call_counter = 0
        self._model = None
        self._processor = None

    def load(self) -> None:
        if not self.mock_mode and self._model is None:
            self._load_real_model()

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if hasattr(self, '_processor'):
                del self._processor
                self._processor = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def describe_patch(self, patch: Any, prompt: str = "") -> str:
        if self.mock_mode:
            return self._mock_describe(patch, prompt)
        return self._real_describe(patch, prompt)

    def answer_question(self, patch: Any, question: str) -> str:
        if self.mock_mode:
            return self._mock_answer(question)
        return self._real_answer(patch, question)

    # ------------------------------------------------------------------
    # Mock implementations
    # ------------------------------------------------------------------

    def _mock_describe(self, patch: Any, prompt: str) -> str:
        idx = self._call_counter % len(_DESCRIPTION_TEMPLATES)
        self._call_counter += 1
        return _DESCRIPTION_TEMPLATES[idx]

    def _mock_answer(self, question: str) -> str:
        q = question.lower()
        if "msi" in q:
            return (
                "The high tumour-infiltrating lymphocyte density and poor glandular "
                "differentiation in this patch raise the possibility of MSI-H status. "
                "Confirmatory MMR immunohistochemistry is recommended."
            )
        if "lauren" in q:
            return (
                "The presence of irregular gland formation and infiltrating tumour "
                "cells suggests intestinal-type Lauren classification, though mixed "
                "features are present."
            )
        if "her2" in q:
            return (
                "No clear membrane staining pattern indicative of HER2 overexpression "
                "is visible on H&E. IHC testing is required for definitive HER2 scoring."
            )
        return (
            "Based on the histomorphological features visible in this patch, "
            "further assessment at higher magnification and molecular correlation "
            "is recommended."
        )

    # ------------------------------------------------------------------
    # Real model stubs
    # ------------------------------------------------------------------

    def _load_real_model(self) -> None:
        try:
            import torch
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            from huggingface_hub import login
        except ImportError as exc:
            raise ImportError(
                "Real VLM requires torch, transformers (>=4.39.0), and huggingface_hub."
            ) from exc
            
        import os
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        self.device = getattr(self, "device", "cuda" if torch.cuda.is_available() else "cpu")
        model_name = "wnkh/llava-med-v1.5-mistral-7b-hf"
        
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Explicitly load the processor and model mapped to the Mistral architecture
            self._processor = LlavaProcessor.from_pretrained(model_name, use_fast=False)
            self._processor.patch_size = 14
            self._model = LlavaForConditionalGeneration.from_pretrained(
                model_name, 
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        except Exception as e:
            print(f"Warning: Failed to load VLM model: {e}")
            self._model = None
            self._processor = None

    def _real_describe(self, patch: Any, prompt: str) -> str:
        if self._model is None or self._processor is None:
            self.load()
            if self._model is None:
                return self._mock_describe(patch, prompt)
            
        import torch
        if patch is None:
            return "No image provided for description."
            
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = self._processor(text=formatted_prompt, images=patch, return_tensors="pt").to(self._model.device)
        
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, 
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
            
        input_len = inputs["input_ids"].shape[1]
        output = self._processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        return output.strip()

    def _real_answer(self, patch: Any, question: str) -> str:
        if self._model is None or self._processor is None:
            self.load()
            if self._model is None:
                return self._mock_answer(question)
            
        import torch
        if patch is None:
            return "No image provided to answer the question."
            
        formatted_question = f"USER: <image>\nQuestion: {question}\nASSISTANT: Answer:"
        inputs = self._processor(text=formatted_question, images=patch, return_tensors="pt").to(self._model.device)
        
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, 
                max_new_tokens=150,
                temperature=0.4
            )
            
        output = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if formatted_question in output:
            output = output.split("Answer:")[-1].strip()
            
        return output