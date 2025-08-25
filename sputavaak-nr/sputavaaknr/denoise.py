import torch
import torchaudio
import ffmpeg
import numpy as np

from df.enhance import enhance, init_df

def read_audio(filename: str, normalize: bool = True) -> tuple[torch.Tensor, int]:
    """
    Reads an audio file (m4a, mp3, wav, etc.) using ffmpeg and returns
    a tensor (channels, samples) and sample rate, like torchaudio.load.

    Args:
        filename (str): Path to input audio file
        normalize (bool): If True, scale to [-1.0, 1.0] float32

    Returns:
        waveform (Tensor): shape (channels, samples), dtype=torch.float32
        sample_rate (int): sample rate of the audio
    """
    # Probe file for sample rate and channels
    info = ffmpeg.probe(filename)
    stream = next(s for s in info['streams'] if s['codec_type'] == 'audio')
    sample_rate = int(stream['sample_rate'])
    channels = int(stream['channels'])

    # Decode to raw float32 PCM via ffmpeg pipe
    out, _ = (
        ffmpeg
        .input(filename)
        .output('pipe:', format='f32le', ac=channels, ar=sample_rate)
        .run(capture_stdout=True, capture_stderr=True, quiet=True)
    )

    # Convert to numpy array
    audio_np = np.frombuffer(out, np.float32).reshape(-1, channels)

    # Convert to torch tensor (channels, samples)
    waveform = torch.from_numpy(audio_np).T

    # Normalize if requested
    if normalize:
        waveform = waveform.clamp(-1.0, 1.0)

    return waveform, sample_rate


def save_mp3(filename: str, waveform: torch.Tensor, sample_rate: int, bitrate: str = "192k"):
    """
    Save a torch audio waveform to an MP3 file using ffmpeg-python.

    Args:
        filename (str): Path to output mp3 file
        waveform (Tensor): shape (channels, samples)
        sample_rate (int): sample rate of the audio
        bitrate (str): mp3 bitrate, e.g. '128k', '192k', '320k'
    """
    if waveform.dim() != 2:
        raise ValueError("Waveform must be 2D (channels, samples).")

    # Convert tensor to numpy float32, channels-last for ffmpeg
    audio_np = waveform.numpy().T.astype(np.float32)

    # Start ffmpeg process
    process = (
        ffmpeg
        .input(
            'pipe:0',
            format='f32le',
            ac=audio_np.shape[1],
            ar=sample_rate
        )
        .output(
            filename,
            format='mp3',
            ac=audio_np.shape[1],
            ar=sample_rate,
            audio_bitrate=bitrate
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )

    # Write PCM buffer to stdin
    process.stdin.write(audio_np.tobytes())
    process.stdin.close()
    process.wait()


def denoise(input :str ="input.mp3", output:str = "output.mp3", bitrate: str = "128k"):

    # Initialize DeepFilterNet
    model, df_state, _ = init_df()

    # Load noisy audio
    # waveform, sr = torchaudio.load(input)
    waveform, sample_rate = read_audio(input)

    # DeepFilterNet models expect 48kHz
    if sample_rate != 48000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 48000)
        sample_rate = 48000

    # Run the enhancement
    print("Enhancing audio...")

    # Note: the enhance function can accept a 1D or 2D tensor, but the internal
    # df_features function requires 2D, which is where the error originated.
    # Passing the correctly shaped tensor here ensures the downstream process works.
    enhanced_audio = enhance(model, df_state, waveform)
    save_mp3(output, enhanced_audio, sample_rate, bitrate)

    print(f"Audio enhanced and saved to {output}")