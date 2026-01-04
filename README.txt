FISH SPEECH TTS
===============

Local voice cloning with the openaudio-s1-mini model. Modified for compatibility 
with macOS Apple Silicon (MPS).

SETUP
-----

cd ~/Development/fish-speech
python3.12 -m venv venv
source venv/bin/activate
pip install -e .
pip install torchcodec

MODEL DOWNLOAD
--------------

Requires Hugging Face authentication:

huggingface-cli login
huggingface-cli download fishaudio/openaudio-s1-mini \
  --local-dir checkpoints/openaudio-s1-mini \
  --local-dir-use-symlinks False

Model size: ~3.3GB (1.6GB model.pth + 1.7GB codec.pth)

API USAGE
---------

Fish Speech provides three main operations for voice cloning:

1. ENCODE VOICE REFERENCE (WAV → PT)
   Pre-encode a voice reference for faster reuse

2. GENERATE VOICE FROM TEXT + REFERENCE PT (FAST)
   Use pre-encoded reference for faster generation

3. GENERATE VOICE FROM TEXT + REFERENCE WAV (DIRECT)
   Generate directly from audio file (encodes on-the-fly)


EXAMPLE 1: Encode Voice Reference
----------------------------------

import sys
import torch
import torchaudio
from pathlib import Path

# Add fish-speech to path
sys.path.insert(0, str(Path.home() / "Development/fish-speech"))
from fish_speech.models.dac.inference import load_model as load_decoder

def encode_reference(audio_file, output_file, device="mps"):
    # Load decoder model
    decoder_model = load_decoder(
        config_name="modded_dac_vq",
        checkpoint_path=str(Path.home() / "Development/fish-speech/checkpoints/openaudio-s1-mini/codec.pth"),
        device=device
    )
    
    # Load and resample audio
    ref_audio, sr = torchaudio.load(audio_file)
    if sr != 44100:
        resampler = torchaudio.transforms.Resample(sr, 44100)
        ref_audio = resampler(ref_audio)
    
    # Encode reference audio
    with torch.no_grad():
        ref_audio = ref_audio.to(device)
        if ref_audio.dim() == 2:
            ref_audio = ref_audio.mean(0, keepdim=True)
        encoded = decoder_model.encode(ref_audio.unsqueeze(0))
        ref_codes = encoded[0][0]  # Shape: (codebook, time)
    
    # Save encoded codes
    torch.save(ref_codes.cpu(), output_file)
    return ref_codes.shape

# Usage
encode_reference("voice.wav", "voice_codes.pt")


EXAMPLE 2: Generate from Text + Reference PT (Fast)
----------------------------------------------------

import sys
import torch
import torchaudio
from pathlib import Path

sys.path.insert(0, str(Path.home() / "Development/fish-speech"))
from fish_speech.models.text2semantic.llama import DualARTransformer
from fish_speech.models.text2semantic.inference import generate_long, init_model
from fish_speech.models.dac.inference import load_model as load_decoder_model

def generate_tts_from_codes(text, reference_codes_file, output_file, device="mps"):
    # Load LLAMA model
    llama_model_path = Path.home() / "Development/fish-speech/checkpoints/openaudio-s1-mini"
    llama_model, decode_one_token = init_model(
        checkpoint_path=str(llama_model_path),
        device=device,
        precision=torch.float16,
        compile=False
    )
    
    # Load decoder model
    decoder_model = load_decoder_model(
        config_name="modded_dac_vq",
        checkpoint_path=str(Path.home() / "Development/fish-speech/checkpoints/openaudio-s1-mini/codec.pth"),
        device=device
    )
    
    # Load pre-encoded reference codes
    ref_codes = torch.load(reference_codes_file, weights_only=True)
    ref_codes = ref_codes.to(device).long()
    
    # Generate speech codes
    codes = []
    for response in generate_long(
        model=llama_model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=1,
        max_new_tokens=2048,
        top_p=0.7,
        repetition_penalty=1.2,
        temperature=0.7,
        prompt_tokens=ref_codes,
        prompt_text=""
    ):
        if response.action == "sample":
            codes.append(response.codes)
    
    # Decode to audio
    if codes:
        codes_tensor = torch.cat(codes, dim=1)
        feature_lengths = torch.tensor([codes_tensor.shape[1]], device=device)
        with torch.no_grad():
            decoded = decoder_model.decode(codes_tensor, feature_lengths)
            audio = decoded[0] if isinstance(decoded, tuple) else decoded
        
        # Save audio
        audio_np = audio.squeeze().cpu().numpy()
        torchaudio.save(output_file, torch.from_numpy(audio_np).unsqueeze(0), 44100)
        return True
    return False

# Usage
generate_tts_from_codes(
    "Hello, this is a test.",
    "voice_codes.pt",
    "output.wav"
)


EXAMPLE 3: Generate from Text + Reference WAV (Direct)
-------------------------------------------------------

Use the same code as Example 2, but encode the reference audio on-the-fly:

# Before loading ref_codes, add this to encode from WAV:
ref_audio, sr = torchaudio.load(reference_audio_file)
if sr != 44100:
    resampler = torchaudio.transforms.Resample(sr, 44100)
    ref_audio = resampler(ref_audio)

with torch.no_grad():
    ref_audio = ref_audio.to(device)
    if ref_audio.dim() == 2:
        ref_audio = ref_audio.mean(0, keepdim=True)
    encoded = decoder_model.encode(ref_audio.unsqueeze(0))
    ref_codes = encoded[0][0]

# Then continue with the same generate_long() code from Example 2


PERFORMANCE
-----------

- Model Loading: ~10 seconds (LLAMA + decoder)
- Reference Encoding: ~2 seconds (if from WAV)
- Token Generation: ~3-4 tokens/second on M4 MPS
- Example: 150 characters → ~40 seconds total generation time


MODIFICATIONS
-------------

This fork includes compatibility fixes for macOS:

- fish_speech/inference_engine/reference_loader.py: 
  Hardcoded ffmpeg backend (torchaudio 2.9+)
  
- tools/run_webui.py: 
  Removed deprecated show_api parameter (Gradio 6.0)


REFERENCE AUDIO GUIDELINES
---------------------------

- Format: WAV, mono or stereo (auto-converted to mono)
- Sample Rate: Any (auto-resampled to 44.1kHz)
- Duration: 5-20 seconds recommended
- Quality: Clean speech, minimal background noise
- Content: Representative of target voice characteristics


LICENSE
-------

Original Fish Speech: BSD-3-Clause
