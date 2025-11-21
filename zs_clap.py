from embedding_index import safe_load_audio, load_clap
import torch

LABELS = [
    "drums",
    "keyboard",
    "synth",
    "piano",
    "bass guitar",
]

processor, model, device = load_clap()

def generate_zero_shot_labels(audio_file):
    y, sr = safe_load_audio(audio_file)
    if y is None:
        return "Unknown"

    inputs_audio = processor(audios=[y], sampling_rate=sr, return_tensors="pt")
    inputs_audio = {k: v.to(device) for k, v in inputs_audio.items()}

    text_inputs = processor(text=LABELS, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        a_emb = model.get_audio_features(**inputs_audio)
        t_emb = model.get_text_features(**text_inputs)

    a_emb = torch.nn.functional.normalize(a_emb, p=2, dim=1)
    t_emb = torch.nn.functional.normalize(t_emb, p=2, dim=1)

    sims = (a_emb @ t_emb.T).cpu().numpy()[0]
    return LABELS[sims.argmax()]
