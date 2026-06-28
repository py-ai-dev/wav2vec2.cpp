#!/usr/bin/env python3
"""
Parity test: compare wav2vec2-cli GGUF output against HuggingFace reference model.

Usage:
    python scripts/parity_test.py \\
        --model jonatasgrosman/wav2vec2-large-xlsr-53-arabic \\
        --gguf /tmp/model_f16.gguf \\
        --lang ar \\
        --cli ./build/wav2vec2-cli \\
        --n-samples 10

Returns exit code 1 if mean CER exceeds --threshold (default 5%).
"""

import argparse, subprocess, sys, os, json, tempfile
import numpy as np
import soundfile as sf

# ── Edit distance (CER/WER) ──────────────────────────────────────────────────

def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1]==b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]

def cer(ref: str, hyp: str) -> float:
    ref = ref.strip(); hyp = hyp.strip()
    if not ref: return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)

def wer(ref: str, hyp: str) -> float:
    r = ref.strip().split(); h = hyp.strip().split()
    if not r: return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)

# ── HF reference ─────────────────────────────────────────────────────────────

def run_hf(model_id: str, wav_paths: list[str]) -> dict[str, str]:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import torch

    proc  = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()

    results = {}
    with torch.no_grad():
        for path in wav_paths:
            audio, sr = sf.read(path, dtype='float32')
            if audio.ndim > 1: audio = audio.mean(axis=1)
            if sr != 16000:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(sr, 16000)
                audio = resample_poly(audio, 16000 // g, sr // g).astype('float32')
            inputs = proc(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            logits = model(**inputs).logits
            ids    = torch.argmax(logits, dim=-1)
            text   = proc.batch_decode(ids)[0]
            results[path] = text.strip()
    return results

# ── CLI reference ─────────────────────────────────────────────────────────────

def run_cli(gguf: str, wav_paths: list[str], cli: str) -> dict[str, str]:
    results = {}
    for path in wav_paths:
        r = subprocess.run([cli, "-m", gguf, "-f", path, "-t", "4"],
                           capture_output=True, text=True)
        results[path] = r.stdout.strip()
    return results

# ── Dataset download ──────────────────────────────────────────────────────────

FLEURS_LANG = {
    'ja': 'ja_jp', 'ar': 'ar_eg', 'pt': 'pt_br', 'ru': 'ru_ru',
    'zh': 'cmn_hans_cn', 'te': 'te_in', 'ta': 'ta_in', 'hi': 'hi_in',
    'bn': 'bn_in', 'ml': 'ml_in', 'sw': 'sw_ke', 'pl': 'pl_pl',
    'el': 'el_gr', 'hu': 'hu_hu', 'nl': 'nl_nl', 'id': 'id_id',
    'ro': 'ro_ro', 'fa': 'fa_ir', 'fi': 'fi_fi', 'en': 'en_us',
}

def download_samples(lang: str, n: int, out_dir: str) -> list[tuple[str, str]]:
    """Return list of (wav_path, reference_text)."""
    import datasets as ds
    os.makedirs(out_dir, exist_ok=True)
    fleurs_code = FLEURS_LANG.get(lang, lang)
    try:
        dataset = ds.load_dataset('google/fleurs', fleurs_code,
                                  split='test', streaming=True, trust_remote_code=True)
    except Exception:
        print(f"  [warn] FLEURS not available for {lang}, trying Common Voice")
        dataset = ds.load_dataset('mozilla-foundation/common_voice_13_0', lang,
                                  split='test', streaming=True, trust_remote_code=True)

    samples = []
    for i, item in enumerate(dataset):
        if i >= n: break
        a = item['audio']
        path = os.path.join(out_dir, f"sample_{i}.wav")
        sf.write(path, np.array(a['array'], dtype='float32'), a['sampling_rate'])
        ref = item.get('transcription') or item.get('sentence') or ''
        samples.append((path, ref.strip()))
    return samples

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',     required=True,  help='HF model ID')
    ap.add_argument('--gguf',      required=True,  help='GGUF file path')
    ap.add_argument('--lang',      default='en',   help='2-letter language code')
    ap.add_argument('--audio-dir', default=None,   help='Use existing WAV files in this dir')
    ap.add_argument('--cli',       default='./build/wav2vec2-cli')
    ap.add_argument('--n-samples', type=int, default=10)
    ap.add_argument('--threshold', type=float, default=0.05, help='Max acceptable mean CER')
    ap.add_argument('--out',       default=None,   help='Save results JSON')
    args = ap.parse_args()

    print(f"\nParity test: {args.model}")
    print(f"  GGUF:    {args.gguf}")
    print(f"  CLI:     {args.cli}")

    # Get audio samples
    if args.audio_dir:
        import glob
        wav_files = sorted(glob.glob(os.path.join(args.audio_dir, '*.wav')))[:args.n_samples]
        samples = [(w, '') for w in wav_files]
        print(f"  Using {len(samples)} WAVs from {args.audio_dir}")
    else:
        tmpdir = tempfile.mkdtemp(prefix=f'wv2parity_{args.lang}_')
        print(f"  Downloading {args.n_samples} FLEURS samples ({args.lang}) → {tmpdir}")
        samples = download_samples(args.lang, args.n_samples, tmpdir)

    wav_paths = [s[0] for s in samples]
    hf_refs   = {s[0]: s[1] for s in samples}   # may be empty if from --audio-dir

    # Run HF model
    print("  Running HF reference model...")
    hf_out = run_hf(args.model, wav_paths)

    # Run CLI
    print("  Running CLI...")
    cli_out = run_cli(args.gguf, wav_paths, args.cli)

    # Compute metrics
    rows = []
    for path in wav_paths:
        hf  = hf_out.get(path, '').upper()
        cli = cli_out.get(path, '').upper()
        ref = hf_refs.get(path, '').upper()
        c   = cer(hf, cli)
        w   = wer(hf, cli)
        rows.append({'file': os.path.basename(path), 'hf': hf, 'cli': cli,
                     'ref': ref, 'cer': c, 'wer': w})

    mean_cer = sum(r['cer'] for r in rows) / len(rows) if rows else 0
    mean_wer = sum(r['wer'] for r in rows) / len(rows) if rows else 0

    # Print table
    print(f"\n{'FILE':<20} {'CER':>6} {'WER':>6}")
    print('-' * 35)
    for r in rows:
        flag = ' *** DIVERGED ***' if r['cer'] > args.threshold else ''
        print(f"{r['file']:<20} {r['cer']:>6.2%} {r['wer']:>6.2%}{flag}")
        if r['cer'] > args.threshold:
            print(f"  HF:  {r['hf'][:120]}")
            print(f"  CLI: {r['cli'][:120]}")
    print('-' * 35)
    print(f"{'MEAN':<20} {mean_cer:>6.2%} {mean_wer:>6.2%}")

    status = 'PASS' if mean_cer <= args.threshold else 'FAIL'
    print(f"\nResult: {status}  (threshold={args.threshold:.0%})")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'model': args.model, 'gguf': args.gguf,
                       'mean_cer': mean_cer, 'mean_wer': mean_wer,
                       'status': status, 'rows': rows}, f, indent=2)
        print(f"Saved: {args.out}")

    return 0 if mean_cer <= args.threshold else 1

if __name__ == '__main__':
    sys.exit(main())
