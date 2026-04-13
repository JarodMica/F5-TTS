import os

from f5_tts.api_legacy import F5TTS as LegacyF5TTS
from f5_tts.api_modern import F5TTSModern


def _normalize_model_name(model_name):
    if not model_name:
        return None
    aliases = {
        "F5-TTS": "OpenF5TTS_v2_Base",
        "F5TTS": "OpenF5TTS_v2_Base",
        "OpenF5-TTS": "OpenF5TTS_v2_Base",
        "OpenF5-TTS-Base": "OpenF5TTS_v2_Base",
        "E2TTS": "E2-TTS",
    }
    return aliases.get(model_name, model_name)


def _resolve_runtime_family(runtime_family, model):
    if runtime_family:
        normalized_runtime = str(runtime_family).strip().lower()
        aliases = {
            "v1": "modern",
            "v1+": "modern",
            "modern": "modern",
            "pre-v1": "legacy",
            "prev1": "legacy",
            "legacy": "legacy",
        }
        normalized_runtime = aliases.get(normalized_runtime, normalized_runtime)
        if normalized_runtime not in {"legacy", "modern"}:
            raise ValueError(f"Unsupported F5 runtime_family: {runtime_family}")
        return normalized_runtime

    normalized_model = _normalize_model_name(model)
    if normalized_model in {"OpenF5TTS_v2_Base", "F5TTS_v1_Base"}:
        return "modern"

    return "modern"


class F5TTS:
    def __init__(
        self,
        model_type="F5-TTS",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_name="vocos",
        vocos_local_path=None,
        model_local_path=None,
        duration_model=True,
        duration_model_path=None,
        device=None,
        model=None,
        model_cfg=None,
        hf_cache_dir=None,
        runtime_family=None,
    ):
        resolved_runtime_family = _resolve_runtime_family(runtime_family, model)
        use_modern = resolved_runtime_family == "modern"
        impl_cls = F5TTSModern if use_modern else LegacyF5TTS
        self._impl = impl_cls(
            model_type=model_type,
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            ode_method=ode_method,
            use_ema=use_ema,
            vocoder_name=vocoder_name,
            vocos_local_path=vocos_local_path,
            model_local_path=model_local_path,
            duration_model=duration_model,
            duration_model_path=duration_model_path,
            device=device,
            model=model,
            model_cfg=model_cfg,
            hf_cache_dir=hf_cache_dir,
            runtime_family=resolved_runtime_family,
        )
        self.runtime_family = resolved_runtime_family

    def __getattr__(self, item):
        return getattr(self._impl, item)

    def infer(self, *args, **kwargs):
        return self._impl.infer(*args, **kwargs)

    def close(self):
        if hasattr(self._impl, "close"):
            self._impl.close()
