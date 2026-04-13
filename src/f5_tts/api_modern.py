import os
import random
import sys

import soundfile as sf
import torch
import tqdm
from cached_path import cached_path

from f5_tts._modern.infer.utils_infer import clear_runtime_caches as clear_modern_runtime_caches
from f5_tts._modern.infer.utils_infer import (
    hop_length,
    infer_process,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    target_sample_rate,
    transcribe,
)
from f5_tts._modern.infer.utils_infer import load_model as load_modern_model
from f5_tts._modern.model import DiT
from f5_tts._modern.model.utils import seed_everything


MODEL_SPECS = {
    "OpenF5TTS_v2_Base": {
        "model_cls": DiT,
        "model_cfg": {
            "dim": 1024,
            "depth": 22,
            "heads": 16,
            "ff_mult": 2,
            "text_dim": 512,
            "text_mask_padding": True,
            "qk_norm": None,
            "conv_layers": 4,
            "pe_attn_head": None,
            "attn_backend": "torch",
            "attn_mask_enabled": False,
            "checkpoint_activations": False,
        },
        "target_sample_rate": 24000,
        "mel_spec_type": "vocos",
        "default_ckpt": "hf://mrfakename/OpenF5-TTS-Base/model.pt",
        "default_vocab": "hf://mrfakename/OpenF5-TTS-Base/vocab.txt",
    },
    "F5TTS_v1_Base": {
        "model_cls": DiT,
        "model_cfg": {
            "dim": 1024,
            "depth": 22,
            "heads": 16,
            "ff_mult": 2,
            "text_dim": 512,
            "text_mask_padding": True,
            "qk_norm": None,
            "conv_layers": 4,
            "pe_attn_head": None,
            "attn_backend": "torch",
            "attn_mask_enabled": False,
            "checkpoint_activations": False,
        },
        "target_sample_rate": 24000,
        "mel_spec_type": "vocos",
        "default_ckpt": "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
        "default_vocab": "",
    },
}


def _normalize_model_name(model_name):
    if not model_name:
        return None
    aliases = {
        "F5-TTS": "OpenF5TTS_v2_Base",
        "F5TTS": "OpenF5TTS_v2_Base",
        "OpenF5-TTS": "OpenF5TTS_v2_Base",
        "OpenF5-TTS-Base": "OpenF5TTS_v2_Base",
    }
    return aliases.get(model_name, model_name)


def _resolve_artifact_path(path_value, cache_dir=None):
    if not path_value:
        return ""
    if str(path_value).startswith("hf://"):
        return str(cached_path(str(path_value), cache_dir=cache_dir))
    return str(path_value)


def _pick_vocoder_paths(vocos_local_path):
    if not vocos_local_path:
        return None, None

    config_path = os.path.join(vocos_local_path, "config.yaml")
    model_path = os.path.join(vocos_local_path, "pytorch_model.bin")
    if os.path.isfile(config_path) and os.path.isfile(model_path):
        return vocos_local_path, None

    return None, vocos_local_path


class F5TTSModern:
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
        del duration_model
        del duration_model_path
        del model_cfg
        del runtime_family

        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.seed = -1
        self.model_name = _normalize_model_name(model) or _normalize_model_name(model_type) or "OpenF5TTS_v2_Base"

        if self.model_name not in MODEL_SPECS:
            raise ValueError(f"Unsupported modern F5 model: {self.model_name}")

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        spec = MODEL_SPECS[self.model_name]
        self.mel_spec_type = spec["mel_spec_type"] if not vocoder_name else vocoder_name

        vocoder_local_dir, vocoder_cache_dir = _pick_vocoder_paths(vocos_local_path)
        self.vocoder = load_vocoder(
            self.mel_spec_type,
            vocoder_local_dir is not None,
            vocoder_local_dir or "",
            self.device,
            hf_cache_dir=vocoder_cache_dir,
        )

        cache_dir = model_local_path or hf_cache_dir
        if not ckpt_file:
            ckpt_file = _resolve_artifact_path(spec["default_ckpt"], cache_dir=cache_dir)
        else:
            ckpt_file = _resolve_artifact_path(ckpt_file, cache_dir=cache_dir)

        if not vocab_file:
            vocab_file = _resolve_artifact_path(spec["default_vocab"], cache_dir=cache_dir)
        else:
            vocab_file = _resolve_artifact_path(vocab_file, cache_dir=cache_dir)

        self.target_sample_rate = spec["target_sample_rate"]
        self.ema_model = load_modern_model(
            spec["model_cls"],
            dict(spec["model_cfg"]),
            ckpt_file,
            self.mel_spec_type,
            vocab_file,
            ode_method,
            use_ema,
            self.device,
        )

    def transcribe(self, ref_audio, language=None):
        return transcribe(ref_audio, language)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)
        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spec, file_spec):
        save_spectrogram(spec, file_spec)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spect=None,
        seed=-1,
    ):
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text, show_info=show_info)

        wav, sr, spec = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spect is not None:
            self.export_spectrogram(spec, file_spect)

        return wav, sr, spec

    def close(self):
        clear_modern_runtime_caches()
        self.vocoder = None
        self.ema_model = None
        self.final_wave = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
