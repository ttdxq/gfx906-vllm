"""Model registry entries imported from the upstream model-support window.

The legacy pre-refactor registry layout is kept, so these entries are
kept separate from the local tables.  The registry merges them after its
legacy entries and preserves the local Qwen3.5/Qwen3.8 mappings.
"""

from __future__ import annotations


MODEL_SUPPORT_ENTRIES: dict[str, tuple[str, str]] = {
    # Text and MoE models.
    "BailingMoeV3ForCausalLM": (
        "vllm.model_executor.models.bailing_moe_v3",
        "BailingMoeV3ForCausalLM",
    ),
    "Cohere2MoeForCausalLM": (
        "vllm.model_executor.models.cohere2_moe",
        "Cohere2MoeForCausalLM",
    ),
    "DeepseekV4ForCausalLM": (
        "vllm.models.deepseek_v4",
        "DeepseekV4ForCausalLM",
    ),
    "Gemma4ForCausalLM": (
        "vllm.model_executor.models.gemma4",
        "Gemma4ForCausalLM",
    ),
    "HrmTextForCausalLM": (
        "vllm.model_executor.models.hrm_text",
        "HrmTextForCausalLM",
    ),
    "HYV3ForCausalLM": (
        "vllm.model_executor.models.hy_v3",
        "HYV3ForCausalLM",
    ),
    "HCXVisionV2ForCausalLM": (
        "vllm.model_executor.models.hyperclovax_vision_v2",
        "HCXVisionV2ForCausalLM",
    ),
    "HyperCLOVAXForCausalLM": (
        "vllm.model_executor.models.hyperclovax",
        "HyperCLOVAXForCausalLM",
    ),
    "KimiLinearForCausalLM": (
        "vllm.models.kimi_k3",
        "KimiLinearForCausalLM",
    ),
    "LagunaForCausalLM": (
        "vllm.model_executor.models.laguna",
        "LagunaForCausalLM",
    ),
    "MiMoV2ForCausalLM": (
        "vllm.model_executor.models.mimo_v2",
        "MiMoV2ForCausalLM",
    ),
    "MiMoV2FlashForCausalLM": (
        "vllm.model_executor.models.mimo_v2",
        "MiMoV2FlashForCausalLM",
    ),
    "MiniMaxM3SparseForCausalLM": (
        "vllm.models.minimax_m3",
        "MiniMaxM3SparseForCausalLM",
    ),
    "MuseGlimmerForCausalLM": (
        "vllm.model_executor.models.muse_glimmer",
        "MuseGlimmerForCausalLM",
    ),
    "Param2MoEForCausalLM": (
        "vllm.model_executor.models.param2moe",
        "Param2MoEForCausalLM",
    ),
    "Rnj1ForCausalLM": (
        "vllm.model_executor.models.rnj1",
        "Rnj1ForCausalLM",
    ),
    "SarvamMoEForCausalLM": (
        "vllm.model_executor.models.sarvam",
        "SarvamMoEForCausalLM",
    ),
    "SarvamMLAForCausalLM": (
        "vllm.model_executor.models.sarvam",
        "SarvamMLAForCausalLM",
    ),
    "TeleChat3ForCausalLM": (
        "vllm.model_executor.models.llama",
        "LlamaForCausalLM",
    ),
    "Ministral3ForCausalLM": (
        "vllm.model_executor.models.mistral",
        "MistralForCausalLM",
    ),

    # Multimodal, vision, audio, and OCR models.
    "Cheers": (
        "vllm.model_executor.models.cheers",
        "CheersForConditionalGeneration",
    ),
    "CheersForConditionalGeneration": (
        "vllm.model_executor.models.cheers",
        "CheersForConditionalGeneration",
    ),
    "CohereAsrForConditionalGeneration": (
        "vllm.model_executor.models.cohere_asr",
        "CohereAsrForConditionalGeneration",
    ),
    "Cosmos3ForConditionalGeneration": (
        "vllm.model_executor.models.cosmos3",
        "Cosmos3ForConditionalGeneration",
    ),
    "Cosmos3EdgeForConditionalGeneration": (
        "vllm.model_executor.models.cosmos3_edge",
        "Cosmos3EdgeForConditionalGeneration",
    ),
    "Dots3NoteForCausalLM": (
        "vllm.models.dots3_note",
        "Dots3NoteForCausalLM",
    ),
    "Exaone4_5_ForConditionalGeneration": (
        "vllm.model_executor.models.exaone4_5",
        "Exaone4_5_ForConditionalGeneration",
    ),
    "Gemma4ForConditionalGeneration": (
        "vllm.model_executor.models.gemma4_mm",
        "Gemma4ForConditionalGeneration",
    ),
    "Gemma4UnifiedForConditionalGeneration": (
        "vllm.model_executor.models.gemma4_unified",
        "Gemma4UnifiedForConditionalGeneration",
    ),
    "Granite4VisionForConditionalGeneration": (
        "vllm.model_executor.models.granite4_vision",
        "Granite4VisionForConditionalGeneration",
    ),
    "GraniteSpeechPlusForConditionalGeneration": (
        "vllm.model_executor.models.granite_speech_plus",
        "GraniteSpeechPlusForConditionalGeneration",
    ),
    "InternS2MobiusForConditionalGeneration": (
        "vllm.model_executor.models.interns2_mobius",
        "InternS2MobiusForConditionalGeneration",
    ),
    "InternS2PreviewForConditionalGeneration": (
        "vllm.model_executor.models.interns2_preview",
        "InternS2PreviewForConditionalGeneration",
    ),
    "KimiAudioForConditionalGeneration": (
        "vllm.model_executor.models.kimi_audio",
        "KimiAudioForConditionalGeneration",
    ),
    "KimiK25ForConditionalGeneration": (
        "vllm.model_executor.models.kimi_k25",
        "KimiK25ForConditionalGeneration",
    ),
    "KimiK3ForConditionalGeneration": (
        "vllm.models.kimi_k3",
        "KimiK3ForConditionalGeneration",
    ),
    "LlavaOnevision2ForConditionalGeneration": (
        "vllm.model_executor.models.llava_onevision2",
        "LlavaOnevision2ForConditionalGeneration",
    ),
    "MiMoV2OmniForCausalLM": (
        "vllm.model_executor.models.mimo_v2_omni",
        "MiMoV2OmniForCausalLM",
    ),
    "MiniCPMV4_6ForConditionalGeneration": (
        "vllm.model_executor.models.minicpmv4_6",
        "MiniCPMV4_6ForConditionalGeneration",
    ),
    "Moondream3ForCausalLM": (
        "vllm.model_executor.models.moondream3",
        "Moondream3ForCausalLM",
    ),
    "MossAudioModel": (
        "vllm.model_executor.models.moss_audio",
        "MossAudioModel",
    ),
    "MossTranscribeDiarizeForConditionalGeneration": (
        "vllm.model_executor.models.moss_transcribe_diarize",
        "MossTranscribeDiarizeForConditionalGeneration",
    ),
    "OpenAIPrivacyFilterForTokenClassification": (
        "vllm.model_executor.models.openai_privacy_filter",
        "OpenAIPrivacyFilterForTokenClassification",
    ),
    "OpenVLAForActionPrediction": (
        "vllm.model_executor.models.openvla",
        "OpenVLAForActionPrediction",
    ),
    "Phi4ForCausalLMV": (
        "vllm.model_executor.models.phi4siglip",
        "Phi4ForCausalLMV",
    ),
    "QianfanOCRForConditionalGeneration": (
        "vllm.model_executor.models.qianfan_ocr",
        "QianfanOCRForConditionalGeneration",
    ),
    "Step3p7ForConditionalGeneration": (
        "vllm.model_executor.models.step3p7",
        "Step3p7ForConditionalGeneration",
    ),
    "UnlimitedOCRForCausalLM": (
        "vllm.model_executor.models.unlimited_ocr",
        "UnlimitedOCRForCausalLM",
    ),

    # Embedding, retrieval, ranking, and token-classification models.
    "ColPaliForRetrieval": (
        "vllm.model_executor.models.colpali",
        "ColPaliModel",
    ),
    "ColQwen3": (
        "vllm.model_executor.models.colqwen3",
        "ColQwen3Model",
    ),
    "ColBERTLfm2Model": (
        "vllm.model_executor.models.colbert",
        "ColBERTLfm2Model",
    ),
    "JinaEmbeddingsV5Model": (
        "vllm.model_executor.models.jina",
        "JinaEmbeddingsV5Model",
    ),
    "JinaForRanking": (
        "vllm.model_executor.models.jina",
        "JinaForRanking",
    ),
    "Qwen3ASRForcedAlignerForTokenClassification": (
        "vllm.model_executor.models.qwen3_asr_forced_aligner",
        "Qwen3ASRForcedAlignerForTokenClassification",
    ),

    # Draft/MTP classes which are not Qwen3.5/Qwen3.8 classes.
    "DFlashDraftModel": (
        "vllm.model_executor.models.qwen3_dflash",
        "DFlashQwen3ForCausalLM",
    ),
    "DFlashLagunaForCausalLM": (
        "vllm.model_executor.models.laguna_dflash",
        "DFlashLagunaForCausalLM",
    ),
    "Eagle3DeepseekV2ForCausalLM": (
        "vllm.model_executor.models.deepseek_eagle3",
        "Eagle3DeepseekV2ForCausalLM",
    ),
    "Eagle3DeepseekV3ForCausalLM": (
        "vllm.model_executor.models.deepseek_eagle3",
        "Eagle3DeepseekV2ForCausalLM",
    ),
    "Eagle3MiniMaxM2ForCausalLM": (
        "vllm.model_executor.models.llama_eagle3",
        "Eagle3LlamaForCausalLM",
    ),
    "Gemma4DSparkModel": (
        "vllm.model_executor.models.gemma4_dspark",
        "Gemma4DSparkForCausalLM",
    ),
    "Gemma4MTPModel": (
        "vllm.model_executor.models.gemma4_mtp",
        "Gemma4MTP",
    ),
    "BailingMoeV3MTPModel": (
        "vllm.model_executor.models.bailing_moe_v3_mtp",
        "BailingMoeV3MTPModel",
    ),
    "Exaone4_5_MTP": (
        "vllm.model_executor.models.exaone4_5_mtp",
        "Exaone4_5_MTP",
    ),
    "HYV3MTPModel": (
        "vllm.model_executor.models.hy_v3_mtp",
        "HYV3MTP",
    ),
    "MiMoV2MTPModel": (
        "vllm.model_executor.models.mimo_v2_mtp",
        "MiMoV2MTP",
    ),
    "MiMoV2OmniMTPModel": (
        "vllm.model_executor.models.mimo_v2_mtp",
        "MiMoV2OmniMTP",
    ),
    "DeepSeekV4MTPModel": (
        "vllm.models.deepseek_v4",
        "DeepSeekV4MTP",
    ),
    "KimiK3MTPModel": (
        "vllm.models.kimi_k3",
        "KimiK3MTP",
    ),
    "MiniMaxM3MTP": (
        "vllm.models.minimax_m3",
        "MiniMaxM3MTP",
    ),
    "Dots3NoteMTPModel": (
        "vllm.models.dots3_note",
        "Dots3NoteMTP",
    ),
    "ExtractHiddenStatesModel": (
        "vllm.model_executor.models.extract_hidden_states",
        "ExtractHiddenStatesModel",
    ),
}

# The entries below are the remaining model architectures from the same
# upstream window.  Keep them separate so the legacy registry table above is
# easy to audit, while still registering every copied implementation.
MODEL_SUPPORT_ENTRIES.update(
    {
        # Additional text and MoE models.
        "AXK1ForCausalLM": (
            "vllm.model_executor.models.AXK1",
            "AXK1ForCausalLM",
        ),
        "BailingMoeV2_5ForCausalLM": (
            "vllm.model_executor.models.bailing_moe_linear",
            "BailingMoeV25ForCausalLM",
        ),
        "BailingMoeV3ForCausalLM": (
            "vllm.model_executor.models.bailing_moe_v3",
            "BailingMoeV3ForCausalLM",
        ),
        "ExaoneMoeForCausalLM": (
            "vllm.model_executor.models.exaone_moe",
            "ExaoneMoeForCausalLM",
        ),
        "Glm4MoeLiteForCausalLM": (
            "vllm.model_executor.models.glm4_moe_lite",
            "Glm4MoeLiteForCausalLM",
        ),
        "IQuestCoderForCausalLM": (
            "vllm.model_executor.models.llama",
            "LlamaForCausalLM",
        ),
        "IQuestLoopCoderForCausalLM": (
            "vllm.model_executor.models.iquest_loopcoder",
            "IQuestLoopCoderForCausalLM",
        ),
        "Jais2ForCausalLM": (
            "vllm.model_executor.models.jais2",
            "Jais2ForCausalLM",
        ),
        "LongcatFlashNgramForCausalLM": (
            "vllm.model_executor.models.longcat_flash_ngram",
            "LongcatFlashNgramForCausalLM",
        ),
        "MellumForCausalLM": (
            "vllm.model_executor.models.mellum",
            "MellumForCausalLM",
        ),
        "NemotronHPuzzleForCausalLM": (
            "vllm.model_executor.models.nemotron_h",
            "NemotronHForCausalLM",
        ),
        "OlmoHybridForCausalLM": (
            "vllm.model_executor.models.olmo_hybrid",
            "OlmoHybridForCausalLM",
        ),
        "Step1ForCausalLM": (
            "vllm.model_executor.models.step1",
            "Step1ForCausalLM",
        ),
        "Step3p5ForCausalLM": (
            "vllm.model_executor.models.step3p5",
            "Step3p5ForCausalLM",
        ),
        # Hardware-isolated model packages.
        "DeepseekV32ForCausalLM": (
            "vllm.models.deepseek_v32",
            "DeepseekV32ForCausalLM",
        ),
        "DeepseekV32MTP": (
            "vllm.models.deepseek_v32",
            "DeepseekV32MTP",
        ),
        "InklingForCausalLM": (
            "vllm.models.inkling",
            "InklingForCausalLM",
        ),
        "InklingForConditionalGeneration": (
            "vllm.models.inkling",
            "InklingForConditionalGeneration",
        ),
        "MiniMaxM3SparseForCausalLM": (
            "vllm.models.minimax_m3",
            "MiniMaxM3SparseForCausalLM",
        ),
        "MiniMaxM3SparseForConditionalGeneration": (
            "vllm.models.minimax_m3",
            "MiniMaxM3SparseForConditionalGeneration",
        ),
        # Embedding, retrieval, ranking, and classification aliases.
        "VoyageQwen3BidirectionalEmbedModel": (
            "vllm.model_executor.models.voyage",
            "VoyageQwen3BidirectionalEmbedModel",
        ),
        "LlamaBidirectionalModel": (
            "vllm.model_executor.models.llama",
            "LlamaBidirectionalModel",
        ),
        "LlamaBidirectionalForSequenceClassification": (
            "vllm.model_executor.models.llama",
            "LlamaBidirectionalForSequenceClassification",
        ),
        "HF_ColBERT": (
            "vllm.model_executor.models.colbert",
            "ColBERTModel",
        ),
        "ColBERTModernBertModel": (
            "vllm.model_executor.models.colbert",
            "ColBERTModernBertModel",
        ),
        "ColBERTJinaRobertaModel": (
            "vllm.model_executor.models.colbert",
            "ColBERTJinaRobertaModel",
        ),
        "ColModernVBertForRetrieval": (
            "vllm.model_executor.models.colmodernvbert",
            "ColModernVBertForRetrieval",
        ),
        "OpsColQwen3Model": (
            "vllm.model_executor.models.colqwen3",
            "ColQwen3Model",
        ),
        "Qwen3VLNemotronEmbedModel": (
            "vllm.model_executor.models.colqwen3",
            "ColQwen3Model",
        ),
        "ModernBertForTokenClassification": (
            "vllm.model_executor.models.modernbert",
            "ModernBertForTokenClassification",
        ),
        "BertForSequenceClassification": (
            "vllm.model_executor.models.bert",
            "BertForSequenceClassification",
        ),
        "GteNewForSequenceClassification": (
            "vllm.model_executor.models.bert_with_rope",
            "GteNewForSequenceClassification",
        ),
        "ModernBertForSequenceClassification": (
            "vllm.model_executor.models.modernbert",
            "ModernBertForSequenceClassification",
        ),
        "JinaVLForRanking": (
            "vllm.model_executor.models.jina_vl",
            "JinaVLForSequenceClassification",
        ),
        # Additional multimodal, audio, and OCR models.
        "AudioFlamingo3ForConditionalGeneration": (
            "vllm.model_executor.models.audioflamingo3",
            "AudioFlamingo3ForConditionalGeneration",
        ),
        "BagelForConditionalGeneration": (
            "vllm.model_executor.models.bagel",
            "BagelForConditionalGeneration",
        ),
        "DeepseekOCR2ForCausalLM": (
            "vllm.model_executor.models.deepseek_ocr2",
            "DeepseekOCR2ForCausalLM",
        ),
        "Eagle2_5_VLForConditionalGeneration": (
            "vllm.model_executor.models.eagle2_5_vl",
            "Eagle2_5_VLForConditionalGeneration",
        ),
        "FireRedASR2ForConditionalGeneration": (
            "vllm.model_executor.models.fireredasr2",
            "FireRedASR2ForConditionalGeneration",
        ),
        "FunASRForConditionalGeneration": (
            "vllm.model_executor.models.funasr",
            "FunASRForConditionalGeneration",
        ),
        "FireRedLIDForConditionalGeneration": (
            "vllm.model_executor.models.fireredlid",
            "FireRedLIDForConditionalGeneration",
        ),
        "FunAudioChatForConditionalGeneration": (
            "vllm.model_executor.models.funaudiochat",
            "FunAudioChatForConditionalGeneration",
        ),
        "DiffusionGemmaForBlockDiffusion": (
            "vllm.model_executor.models.diffusion_gemma",
            "DiffusionGemmaForConditionalGeneration",
        ),
        "GlmAsrForConditionalGeneration": (
            "vllm.model_executor.models.glmasr",
            "GlmAsrForConditionalGeneration",
        ),
        "GlmOcrForConditionalGeneration": (
            "vllm.model_executor.models.glm_ocr",
            "GlmOcrForConditionalGeneration",
        ),
        "InternS1ProForConditionalGeneration": (
            "vllm.model_executor.models.interns1_pro",
            "InternS1ProForConditionalGeneration",
        ),
        "IsaacForConditionalGeneration": (
            "vllm.model_executor.models.isaac",
            "IsaacForConditionalGeneration",
        ),
        "KananaVForConditionalGeneration": (
            "vllm.model_executor.models.kanana_v",
            "KananaVForConditionalGeneration",
        ),
        "MoonshotKimiaForCausalLM": (
            "vllm.model_executor.models.kimi_audio",
            "KimiAudioForConditionalGeneration",
        ),
        "Lfm2VlForConditionalGeneration": (
            "vllm.model_executor.models.lfm2_vl",
            "Lfm2VLForConditionalGeneration",
        ),
        "Molmo2ForConditionalGeneration": (
            "vllm.model_executor.models.molmo2",
            "Molmo2ForConditionalGeneration",
        ),
        "MuseGlimmerForConditionalGeneration": (
            "vllm.model_executor.models.muse_glimmer",
            "MuseGlimmerForCausalLM",
        ),
        "HfMoondream": (
            "vllm.model_executor.models.moondream3",
            "Moondream3ForCausalLM",
        ),
        "NemotronH_Nano_Omni_Reasoning_V3": (
            "vllm.model_executor.models.nano_nemotron_vl",
            "NemotronH_Nano_VL_V2",
        ),
        "NemotronH_Super_Omni_Reasoning_V3": (
            "vllm.model_executor.models.nano_nemotron_vl",
            "NemotronH_Nano_VL_V2",
        ),
        "OpenPanguVLForConditionalGeneration": (
            "vllm.model_executor.models.openpangu_vl",
            "OpenPanguVLForConditionalGeneration",
        ),
        "Ovis2_6ForCausalLM": (
            "vllm.model_executor.models.ovis2_5",
            "Ovis2_5",
        ),
        "Ovis2_6_MoeForCausalLM": (
            "vllm.model_executor.models.ovis2_5",
            "Ovis2_5",
        ),
        "Qwen3ASRForConditionalGeneration": (
            "vllm.model_executor.models.qwen3_asr",
            "Qwen3ASRForConditionalGeneration",
        ),
        "Qwen3ASRRealtimeGeneration": (
            "vllm.model_executor.models.qwen3_asr_realtime",
            "Qwen3ASRRealtimeGeneration",
        ),
        "StepVLForConditionalGeneration": (
            "vllm.model_executor.models.step_vl",
            "StepVLForConditionalGeneration",
        ),
        "VoxtralRealtimeGeneration": (
            "vllm.model_executor.models.voxtral_realtime",
            "VoxtralRealtimeGeneration",
        ),
        "NemotronParseForConditionalGeneration": (
            "vllm.model_executor.models.nemotron_parse",
            "NemotronParseForConditionalGeneration",
        ),
        "VaultGemmaForCausalLM": (
            "transformers",
            "TransformersForCausalLM",
        ),
        "VibeVoiceAsrForConditionalGeneration": (
            "transformers",
            "TransformersMultiModalForCausalLM",
        ),
        # Additional speculative decoding and MTP aliases.
        "MiMoV2MTPModel": (
            "vllm.model_executor.models.mimo_v2_mtp",
            "MiMoV2MTP",
        ),
        "MiMoV2OmniMTPModel": (
            "vllm.model_executor.models.mimo_v2_mtp",
            "MiMoV2OmniMTP",
        ),
        "EagleCohereForCausalLM": (
            "vllm.model_executor.models.cohere_eagle",
            "EagleCohereForCausalLM",
        ),
        "MuseGlimmerAssistantModel": (
            "vllm.model_executor.models.qwen3_dflash",
            "DFlashQwen3ForCausalLM",
        ),
        "DFlashMuseGlimmerAssistantModel": (
            "vllm.model_executor.models.qwen3_dflash",
            "DFlashQwen3ForCausalLM",
        ),
        "DSparkDraftModel": (
            "vllm.models.deepseek_v4",
            "DSparkDeepseekV4ForCausalLM",
        ),
        "Qwen3DSparkModel": (
            "vllm.model_executor.models.qwen3_dspark",
            "Qwen3DSparkForCausalLM",
        ),
        "K3DSparkModel": (
            "vllm.models.kimi_k3.nvidia.dspark_mla",
            "K3DSparkForCausalLM",
        ),
        "PEagleDraftModel": (
            "vllm.model_executor.models.llama_eagle3",
            "Eagle3LlamaForCausalLM",
        ),
        "PeagleLlamaForCausalLM": (
            "vllm.model_executor.models.llama_eagle3",
            "Eagle3LlamaForCausalLM",
        ),
        "Eagle3MiniMaxM2ForCausalLM": (
            "vllm.model_executor.models.llama_eagle3",
            "Eagle3LlamaForCausalLM",
        ),
        "Eagle3Qwen3ForCausalLM": (
            "vllm.model_executor.models.qwen3_eagle3",
            "Eagle3Qwen3ForCausalLM",
        ),
        "PeagleQwen3ForCausalLM": (
            "vllm.model_executor.models.qwen3_eagle3",
            "Eagle3Qwen3ForCausalLM",
        ),
        "EagleMistralForCausalLM": (
            "vllm.model_executor.models.mistral_eagle",
            "EagleMistralForCausalLM",
        ),
        "BailingMoeV25MTPModel": (
            "vllm.model_executor.models.bailing_moe_mtp",
            "BailingMoeV25MTPModel",
        ),
        "InklingMTPModel": (
            "vllm.models.inkling",
            "InklingMTP",
        ),
        "ExaoneMoeMTP": (
            "vllm.model_executor.models.exaone_moe_mtp",
            "ExaoneMoeMTP",
        ),
        "NemotronHMTPModel": (
            "vllm.model_executor.models.nemotron_h_mtp",
            "NemotronHMTP",
        ),
        "Glm4MoeLiteMTPModel": (
            "vllm.model_executor.models.glm4_moe_lite_mtp",
            "Glm4MoeLiteMTP",
        ),
        "GlmOcrMTPModel": (
            "vllm.model_executor.models.glm_ocr_mtp",
            "GlmOcrMTP",
        ),
        "Step3p5MTP": (
            "vllm.model_executor.models.step3p5_mtp",
            "Step3p5MTP",
        ),
        "InternS2MobiusMTP": (
            "vllm.model_executor.models.interns2_mobius",
            "InternS2MobiusMTP",
        ),
        "HYV3MTPModel": (
            "vllm.model_executor.models.hy_v3_mtp",
            "HYV3MTP",
        ),
        "KimiK3MTPModel": (
            "vllm.models.kimi_k3",
            "KimiK3MTP",
        ),
    }
)
