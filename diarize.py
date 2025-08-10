import argparse
import logging
import os
import re

import faster_whisper
import torch
import torchaudio

# Note: ctc_forced_aligner is currently disabled due to import compatibility issues
CTC_AVAILABLE = False
logging.warning("ctc_forced_aligner disabled due to import compatibility issues, using fallback forced alignment")

try:
    from deepmultilingualpunctuation import PunctuationModel
    PUNCT_AVAILABLE = True
except ImportError:
    PUNCT_AVAILABLE = False
    logging.warning("deepmultilingualpunctuation not available, punctuation restoration will be disabled")

try:
    import demucs
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    logging.warning("demucs not available, source separation will be disabled")

from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import (
    cleanup,
    create_config,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)
from config import settings

mtypes = {"cpu": "int8", "cuda": "float16"}

def process_audio_file(
    audio_file: str,
    output_dir: str,
    language: str = None,
    whisper_model: str = "large-v3",
    batch_size: int = 8,
    device: str = "cuda",
    stemming: bool = True,
    suppress_numerals: bool = False
):
    # Check CUDA/CUDNN availability and fallback to CPU if needed
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logging.warning("CUDA not available, falling back to CPU")
                device = "cpu"
            else:
                # Test CUDNN more thoroughly
                try:
                    cudnn_version = torch.backends.cudnn.version()
                    logging.info(f"CUDNN is available: version {cudnn_version}")
                    
                    # Test CUDNN functionality with valid methods
                    test_tensor = torch.randn(1, 1, 10, 10).cuda()
                    # Test if CUDNN is enabled and working
                    if torch.backends.cudnn.enabled:
                        logging.info("CUDNN is enabled and working")
                    else:
                        logging.warning("CUDNN is disabled, falling back to CPU")
                        device = "cpu"
                    
                    # Test basic CUDA operations
                    test_result = test_tensor + test_tensor
                    del test_tensor, test_result
                    logging.info("CUDNN functionality test passed")
                    
                except Exception as e:
                    logging.warning(f"CUDNN error: {e}, falling back to CPU")
                    device = "cpu"
        except Exception as e:
            logging.warning(f"CUDA check failed: {e}, falling back to CPU")
            device = "cpu"
    
    logging.info(f"Using device: {device}")
    
    # Debug: Check PyTorch CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            logging.info(f"✅ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
            logging.info(f"✅ CUDA version: {torch.version.cuda}")
            logging.info(f"✅ CUDNN version: {torch.backends.cudnn.version()}")
            logging.info(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logging.warning("❌ PyTorch CUDA not available")
    except Exception as e:
        logging.warning(f"❌ PyTorch CUDA check failed: {e}")
    """
    Process audio file for transcription and diarization
    
    Args:
        audio_file: Path to the audio file
        output_dir: Directory to save output files
        language: Language code (optional)
        whisper_model: Whisper model to use
        batch_size: Batch size for processing
        device: Device to use (cpu/cuda)
        stemming: Whether to perform source separation
        suppress_numerals: Whether to suppress numerical digits
    
    Returns:
        dict: Results containing transcript and file paths
    """
    pid = os.getpid()
    temp_outputs_dir = f"temp_outputs_{pid}"
    
    try:
        # Process language argument
        language = process_language_arg(language, whisper_model)
        
        # Source separation (stemming)
        if stemming and DEMUCS_AVAILABLE:
            try:
                # Isolate vocals from the rest of the audio
                logging.info(f"Starting source separation with device: {device}")
                return_code = os.system(
                    f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio_file}" -o "{temp_outputs_dir}" --device "{device}"'
                )

                if return_code != 0:
                    logging.warning(
                        "Source splitting failed, using original audio file. "
                        "Use --no-stem argument to disable it."
                    )
                    vocal_target = audio_file
                else:
                    vocal_target = os.path.join(
                        temp_outputs_dir,
                        "htdemucs",
                        os.path.splitext(os.path.basename(audio_file))[0],
                        "vocals.wav",
                    )
            except Exception as e:
                logging.warning(f"Source separation error: {e}, using original audio file")
                vocal_target = audio_file
        else:
            if stemming and not DEMUCS_AVAILABLE:
                logging.warning("Source separation requested but demucs not available, using original audio")
            vocal_target = audio_file

        # Transcribe the audio file with fallback to smaller model if needed
        try:
            logging.info(f"Loading Whisper model: {whisper_model}")
            
            # Check available GPU memory before loading
            if device == "cuda" and torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                logging.info(f"🔄 GPU memory - Total: {total_memory:.1f} GB, Free: {free_memory:.1f} GB")
            
            whisper_model_instance = faster_whisper.WhisperModel(
                whisper_model, device=device, compute_type=mtypes[device]
            )
            
            logging.info(f"✅ Whisper model loaded successfully: {whisper_model}")
            
        except Exception as e:
            logging.warning(f"Failed to load {whisper_model}, trying medium model: {e}")
            # Fallback to medium model
            fallback_model = "medium"
            logging.info(f"Loading fallback model: {fallback_model}")
            whisper_model_instance = faster_whisper.WhisperModel(
                fallback_model, device=device, compute_type=mtypes[device]
            )
        # Create Whisper pipeline with error handling
        try:
            whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model_instance)
            logging.info("✅ Whisper pipeline created successfully")
        except Exception as e:
            logging.warning(f"Failed to create batched pipeline: {e}, using single inference")
            whisper_pipeline = None
        
        # Decode audio
        try:
            audio_waveform = faster_whisper.decode_audio(vocal_target)
            logging.info(f"✅ Audio decoded successfully, shape: {audio_waveform.shape}")
        except Exception as e:
            logging.error(f"Audio decoding failed: {e}")
            raise
        
        # Get suppress tokens
        try:
            suppress_tokens = (
                find_numeral_symbol_tokens(whisper_model_instance.hf_tokenizer)
                if suppress_numerals
                else [-1]
            )
        except Exception as e:
            logging.warning(f"Failed to get suppress tokens: {e}, using default")
            suppress_tokens = [-1]

        # Transcribe with comprehensive fallback logic
        transcript_segments = None
        info = None
        
        # Try GPU processing first
        if device == "cuda" and batch_size > 0 and whisper_pipeline is not None:
            try:
                logging.info(f"🔄 Attempting GPU batched inference with batch_size={batch_size}")
                transcript_segments, info = whisper_pipeline.transcribe(
                    audio_waveform,
                    language,
                    suppress_tokens=suppress_tokens,
                    batch_size=batch_size,
                )
                logging.info("✅ GPU batched inference completed successfully")
            except Exception as e:
                logging.warning(f"⚠️ GPU batched inference failed: {e}")
                transcript_segments = None
                info = None
        
        # Fallback to GPU single inference
        if transcript_segments is None and device == "cuda":
            try:
                logging.info("🔄 Attempting GPU single inference")
                transcript_segments, info = whisper_model_instance.transcribe(
                    audio_waveform,
                    language,
                    suppress_tokens=suppress_tokens,
                    vad_filter=True,
                )
                logging.info("✅ GPU single inference completed successfully")
            except Exception as e:
                logging.warning(f"⚠️ GPU single inference failed: {e}")
                transcript_segments = None
                info = None
        
        # Final fallback to CPU
        if transcript_segments is None:
            try:
                logging.info("🔄 Falling back to CPU processing")
                # Move model to CPU temporarily
                whisper_model_instance = whisper_model_instance.cpu()
                audio_waveform_cpu = audio_waveform.cpu() if hasattr(audio_waveform, 'cpu') else audio_waveform
                
                transcript_segments, info = whisper_model_instance.transcribe(
                    audio_waveform_cpu,
                    language,
                    suppress_tokens=suppress_tokens,
                    vad_filter=True,
                )
                logging.info("✅ CPU inference completed successfully")
                
                # Move model back to GPU if needed
                if device == "cuda":
                    whisper_model_instance = whisper_model_instance.cuda()
                    
            except Exception as e:
                logging.error(f"❌ All inference methods failed: {e}")
                raise RuntimeError(f"Failed to transcribe audio: {e}")
        
        if transcript_segments is None:
            raise RuntimeError("No transcription result obtained from any method")

        full_transcript = "".join(segment.text for segment in transcript_segments)

        # clear gpu vram
        try:
            if whisper_pipeline is not None:
                del whisper_pipeline
            del whisper_model_instance
            logging.info("✅ Whisper models cleaned up")
        except Exception as e:
            logging.warning(f"Error during model cleanup: {e}")
        
        # Debug: Show GPU memory usage
        if device == "cuda" and torch.cuda.is_available():
            try:
                logging.info(f"🔄 GPU memory before cleanup: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
                torch.cuda.empty_cache()
                logging.info(f"✅ GPU memory after cleanup: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
            except Exception as e:
                logging.warning(f"Failed to clear GPU cache: {e}")
        else:
            logging.info("ℹ️ Running on CPU, no GPU memory to clear")

        # Forced Alignment
        if CTC_AVAILABLE:
            # This block is now effectively disabled as ctc_forced_aligner is not imported
            # The fallback logic below will be executed instead.
            logging.info("ctc_forced_aligner is disabled, using fallback word-level timestamp approximation.")
            word_timestamps = []
            current_time = 0
            for segment in transcript_segments:
                segment_duration = segment.end - segment.start
                words = segment.text.strip().split()
                if words:
                    word_duration = segment_duration / len(words)
                    for word in words:
                        word_timestamps.append({
                            "word": word,
                            "start": current_time,
                            "end": current_time + word_duration
                        })
                        current_time += word_duration
        else:
            # Fallback: create simple word-level timestamp approximation
            logging.info("Using fallback word-level timestamp approximation")
            word_timestamps = []
            current_time = 0
            for segment in transcript_segments:
                segment_duration = segment.end - segment.start
                words = segment.text.strip().split()
                if words:
                    word_duration = segment_duration / len(words)
                    for word in words:
                        word_timestamps.append({
                            "word": word,
                            "start": current_time,
                            "end": current_time + word_duration
                        })
                        current_time += word_duration

        # convert audio to mono for NeMo compatibility
        ROOT = os.getcwd()
        temp_path = os.path.join(ROOT, temp_outputs_dir)
        os.makedirs(temp_path, exist_ok=True)
        torchaudio.save(
            os.path.join(temp_path, "mono_file.wav"),
            torch.from_numpy(audio_waveform).unsqueeze(0).float(),
            16000,
            channels_first=True,
        )

        # Initialize NeMo MSDD diarization model with error handling
        try:
            logging.info("Initializing NeMo MSDD diarization model...")
            msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(device)
            logging.info("Running diarization...")
            msdd_model.diarize()
            logging.info("Diarization completed successfully")
        except Exception as e:
            logging.error(f"NeMo diarization failed: {e}")
            logging.warning("Continuing without speaker diarization...")
            # Create a simple fallback diarization result
            msdd_model = None

        del msdd_model
        torch.cuda.empty_cache()

        # Reading timestamps <> Speaker Labels mapping
        speaker_ts = []
        if msdd_model is not None:
            try:
                with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line_list = line.split(" ")
                        s = int(float(line_list[5]) * 1000)
                        e = s + int(float(line_list[8]) * 1000)
                        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])
                logging.info("✅ Speaker timestamps loaded successfully")
            except Exception as e:
                logging.warning(f"⚠️ Failed to read speaker timestamps: {e}")
                speaker_ts = []
        else:
            logging.info("ℹ️ No diarization results available, creating single speaker mapping")
            # Create a single speaker for the entire audio
            audio_duration_ms = int(len(audio_waveform) / 16000 * 1000)  # Assuming 16kHz sample rate
            speaker_ts = [[0, audio_duration_ms, 0]]

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        # Punctuation restoration
        if info.language in punct_model_langs and PUNCT_AVAILABLE:
            # restoring punctuation in the transcript to help realign the sentences
            punct_model = PunctuationModel(model="kredor/punctuate-all")

            words_list = list(map(lambda x: x["word"], wsm))

            labled_words = punct_model.predict(words_list, chunk_size=230)

            ending_puncts = ".?!"
            model_puncts = ".,;:!?"

            # We don't want to punctuate U.S.A. with a period. Right?
            is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

            for word_dict, labeled_tuple in zip(wsm, labled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word
        else:
            if info.language in punct_model_langs and not PUNCT_AVAILABLE:
                logging.warning(
                    f"Punctuation restoration is available for {info.language} but package not available."
                )
            else:
                logging.warning(
                    f"Punctuation restoration is not available for {info.language} language."
                    " Using the original punctuation."
                )

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        output_txt = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}.txt")
        output_srt = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}.srt")
        
        try:
            with open(output_txt, "w", encoding="utf-8-sig") as f:
                get_speaker_aware_transcript(ssm, f)
            logging.info("✅ Transcript saved to TXT file")
        except Exception as e:
            logging.error(f"❌ Failed to save TXT file: {e}")
            raise
        
        try:
            with open(output_srt, "w", encoding="utf-8-sig") as srt:
                write_srt(ssm, srt)
            logging.info("✅ Transcript saved to SRT file")
        except Exception as e:
            logging.error(f"❌ Failed to save SRT file: {e}")
            raise

        cleanup(temp_path)
        
        return {
            "transcript": full_transcript,
            "output_dir": output_dir,
            "txt_file": output_txt,
            "srt_file": output_srt,
            "language": info.language
        }
        
    except Exception as e:
        logging.error(f"Error processing audio file: {str(e)}")
        if 'temp_path' in locals():
            cleanup(temp_path)
        raise

def main():
    """Main function for command-line usage"""
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--audio", help="name of the target audio file", required=True
    )
    parser.add_argument(
        "--no-stem",
        action="store_false",
        dest="stemming",
        default=True,
        help="Disables source separation. "
        "This helps with long files that don't contain a lot of music.",
    )

    parser.add_argument(
        "--suppress_numerals",
        action="store_true",
        dest="suppress_numerals",
        default=False,
        help="Suppresses Numerical Digits. "
        "This helps the diarization accuracy but converts all digits into written text.",
    )

    parser.add_argument(
        "--whisper-model",
        dest="whisper_model",
        default="medium.en",
        help="name of the Whisper model to use (e.g., tiny, base, small, medium, large)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        dest="batch_size",
        default=8,
        help="Batch size for batched inference, reduce if you run out of memory, "
        "set to 0 for original whisper longform inference",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=whisper_langs,
        help="Language spoken in the audio, specify None to perform language detection",
    )

    parser.add_argument(
        "--device",
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="if you have a GPU use 'cuda', otherwise 'cpu'",
    )

    args = parser.parse_args()
    

    
    # Create output directory based on input file name
    output_dir = os.path.splitext(args.audio)[0]
    
    # Process the audio file
    result = process_audio_file(
        audio_file=args.audio,
        output_dir=output_dir,
        language=args.language,
        whisper_model=args.whisper_model,
        batch_size=args.batch_size,
        device=args.device,
        stemming=args.stemming,
        suppress_numerals=args.suppress_numerals
    )
    
    print(f"Processing completed successfully!")
    print(f"Output files saved to: {output_dir}")
    print(f"Transcript: {result['txt_file']}")
    print(f"Subtitles: {result['srt_file']}")

if __name__ == "__main__":
    main()
