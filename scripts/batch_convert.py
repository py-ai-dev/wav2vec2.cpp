#!/usr/bin/env python3
"""
Batch convert wav2vec2 models to GGUF, run parity checks, and push to HuggingFace.

Usage:
    python scripts/batch_convert.py \
        --cli ./build/wav2vec2-cli \
        --org liodon-ai \
        --models-file scripts/target_models.txt  # optional; else uses built-in list
        --skip-parity     # skip parity testing (faster)
        --dry-run         # convert + test but don't push
        --only facebook/wav2vec2-base-960h  # convert one model only

Output structure per model:
    /tmp/wv2_batch/{safe_name}/
        model_f16.gguf
        model_q8_0.gguf
        model_q4_0.gguf
        parity_f16.json
        parity_q8_0.json
        parity_q4_0.json
        README.md
"""

import argparse, subprocess, sys, os, json, re, tempfile, time
from pathlib import Path

# ── Target model registry ─────────────────────────────────────────────────────

MODELS = [
    # (hf_model_id, lang_code, architecture, test_dataset_hint)
    ("facebook/wav2vec2-base-960h",                        "en", "base-960h",         "LibriSpeech"),
    ("facebook/wav2vec2-large-960h",                       "en", "large-960h",         "LibriSpeech"),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-japanese",     "ja", "XLS-R",              "FLEURS ja_jp"),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-polish",       "pl", "XLS-R",              "FLEURS pl_pl"),
    ("indonesian-nlp/wav2vec2-indonesian-javanese-sundanese","id","XLS-R",             "FLEURS id_id"),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-dutch",        "nl", "XLS-R",              "FLEURS nl_nl"),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-greek",        "el", "XLS-R",              "FLEURS el_gr"),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-arabic",       "ar", "XLS-R",              "FLEURS ar_eg"),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",    "hu", "XLS-R",              "FLEURS hu_hu"),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",   "pt", "XLS-R",              "FLEURS pt_br"),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-russian",      "ru", "XLS-R",              "FLEURS ru_ru"),
    ("anuragshas/wav2vec2-large-xlsr-53-telugu",           "te", "XLS-R",              "FLEURS te_in"),
    ("Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",      "ta", "XLS-R",              "FLEURS ta_in"),
    ("theainerd/Wav2Vec2-large-xlsr-hindi",                "hi", "XLS-R",              "FLEURS hi_in"),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn","zh", "XLS-R",              "FLEURS cmn_hans_cn"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_name(model_id: str) -> str:
    return re.sub(r'[^A-Za-z0-9_-]', '_', model_id.split('/')[-1])

def repo_name(model_id: str, org: str) -> str:
    return f"{org}/{model_id.split('/')[-1]}-GGUF"

def file_mb(path: str) -> str:
    try:
        return f"{Path(path).stat().st_size / 1e6:.0f} MB"
    except FileNotFoundError:
        return "N/A"

def run(cmd, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, **kwargs)

def convert_model(model_id: str, out_dir: str, dtype: str) -> str:
    """Convert a model to GGUF and return the output path."""
    out_path = os.path.join(out_dir, f"model_{dtype}.gguf")
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists")
        return out_path
    script = Path(__file__).parent / "convert_to_gguf.py"
    r = run([sys.executable, str(script), model_id, out_path, "--dtype", dtype],
            capture_output=False)
    if r.returncode != 0:
        raise RuntimeError(f"Conversion failed: {model_id} --dtype {dtype}")
    return out_path

def run_parity(model_id: str, gguf: str, lang: str, cli: str, n_samples: int,
               out_json: str, skip_parity: bool) -> dict:
    if skip_parity:
        return {"model": model_id, "gguf": gguf, "mean_cer": 0.0,
                "mean_wer": 0.0, "status": "SKIPPED", "rows": []}

    if os.path.exists(out_json):
        print(f"  [skip] parity already done: {out_json}")
        with open(out_json) as f:
            return json.load(f)

    script = Path(__file__).parent / "parity_test.py"
    r = run([sys.executable, str(script),
             "--model", model_id, "--gguf", gguf,
             "--lang", lang, "--cli", cli,
             "--n-samples", str(n_samples), "--out", out_json],
            capture_output=False)
    if r.returncode not in (0, 1):   # 0=PASS, 1=FAIL (>5% CER), 2+=crash
        raise RuntimeError(f"Parity script crashed for {model_id}")
    with open(out_json) as f:
        return json.load(f)

def generate_model_card(model_id: str, out_dir: str, org: str,
                        arch: str, lang: str, test_dataset: str,
                        parity: dict[str, dict]) -> str:
    template_path = Path(__file__).parent / "model_card_template.md"
    template = template_path.read_text()

    sn    = model_id.split('/')[-1]
    rname = repo_name(model_id, org)

    def pstats(dtype):
        p = parity.get(dtype, {})
        cer = p.get('mean_cer', 0.0)
        wer = p.get('mean_wer', 0.0)
        status = p.get('status', 'SKIPPED')
        return cer, wer, status

    cer_f16,  wer_f16,  st_f16  = pstats("f16")
    cer_q8_0, wer_q8_0, st_q8_0 = pstats("q8_0")
    cer_q4_0, wer_q4_0, st_q4_0 = pstats("q4_0")

    n_samples = max(
        len(parity.get("f16", {}).get("rows", [])),
        len(parity.get("q8_0", {}).get("rows", [])),
        10
    )

    readme = template.format(
        repo_name=sn,
        repo_full_name=rname,
        source_model=model_id,
        size_f16=file_mb(os.path.join(out_dir, "model_f16.gguf")),
        size_q8_0=file_mb(os.path.join(out_dir, "model_q8_0.gguf")),
        size_q4_0=file_mb(os.path.join(out_dir, "model_q4_0.gguf")),
        n_samples=n_samples,
        test_dataset=test_dataset,
        cer_f16=cer_f16,   wer_f16=wer_f16,   status_f16=st_f16,
        cer_q8_0=cer_q8_0, wer_q8_0=wer_q8_0, status_q8_0=st_q8_0,
        cer_q4_0=cer_q4_0, wer_q4_0=wer_q4_0, status_q4_0=st_q4_0,
        architecture=arch,
        language=lang,
        source_license_url=f"https://huggingface.co/{model_id}/blob/main/LICENSE",
    )
    out_path = os.path.join(out_dir, "README.md")
    with open(out_path, 'w') as f:
        f.write(readme)
    return out_path

def push_to_hf(out_dir: str, repo_id: str, dry_run: bool):
    from huggingface_hub import HfApi
    api = HfApi()
    if dry_run:
        print(f"  [dry-run] would push {out_dir} → {repo_id}")
        return
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"  Repo {repo_id} already exists")
    except Exception:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"  Created repo: {repo_id}")

    for fname in ["model_f16.gguf", "model_q8_0.gguf", "model_q4_0.gguf", "README.md"]:
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [warn] missing {fname}, skipping")
            continue
        print(f"  Uploading {fname} ...")
        api.upload_file(path_or_fileobj=fpath,
                        path_in_repo=fname,
                        repo_id=repo_id,
                        repo_type="model")
    print(f"  Pushed: https://huggingface.co/{repo_id}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cli',          default='./build/wav2vec2-cli')
    ap.add_argument('--org',          default='liodon-ai')
    ap.add_argument('--models-file',  default=None)
    ap.add_argument('--only',         default=None, help='Process only this model ID')
    ap.add_argument('--out-dir',      default=os.path.join(tempfile.gettempdir(), 'wv2_batch'))
    ap.add_argument('--n-samples',    type=int, default=10)
    ap.add_argument('--skip-parity',  action='store_true')
    ap.add_argument('--dry-run',      action='store_true')
    args = ap.parse_args()

    models = MODELS
    if args.models_file:
        models = []
        for line in open(args.models_file):
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('\t')
                models.append(tuple(parts))

    if args.only:
        models = [m for m in models if m[0] == args.only]
        if not models:
            print(f"Model {args.only!r} not found in registry.")
            return 1

    print(f"Batch converting {len(models)} models → {args.org}")
    print(f"  Output dir: {args.out_dir}")
    print(f"  CLI:        {args.cli}")
    print(f"  Parity:     {'skipped' if args.skip_parity else f'{args.n_samples} samples/model'}")
    print(f"  Push:       {'DRY RUN' if args.dry_run else 'YES'}\n")

    summary = []

    for model_id, lang, arch, test_dataset in models:
        sn = safe_name(model_id)
        out_dir = os.path.join(args.out_dir, sn)
        os.makedirs(out_dir, exist_ok=True)
        rname = repo_name(model_id, args.org)

        print(f"\n{'='*70}")
        print(f"Model: {model_id}")
        print(f"Repo:  {rname}")
        print(f"{'='*70}")

        row = {"model": model_id, "repo": rname, "error": None,
               "cer_f16": None, "cer_q8_0": None, "cer_q4_0": None}

        try:
            # Step 1: Convert all dtypes
            for dtype in ["f16", "q8_0", "q4_0"]:
                print(f"\n[1/{dtype}] Converting to {dtype.upper()}...")
                convert_model(model_id, out_dir, dtype)

            # Step 2: Parity checks
            parity = {}
            for dtype in ["f16", "q8_0", "q4_0"]:
                print(f"\n[2/{dtype}] Parity check ({dtype.upper()})...")
                gguf = os.path.join(out_dir, f"model_{dtype}.gguf")
                pjson = os.path.join(out_dir, f"parity_{dtype}.json")
                p = run_parity(model_id, gguf, lang, args.cli,
                               args.n_samples, pjson, args.skip_parity)
                parity[dtype] = p
                cer = p.get('mean_cer', 0.0)
                status = p.get('status', '?')
                row[f"cer_{dtype}"] = cer
                print(f"  {dtype.upper()}: CER={cer:.2%}  [{status}]")
                if status == 'FAIL':
                    print(f"  [warn] CER exceeds 5% threshold — check parity_{dtype}.json")

            # Step 3: Model card
            print(f"\n[3] Generating model card...")
            generate_model_card(model_id, out_dir, args.org,
                                arch, lang, test_dataset, parity)
            print(f"  Written: {out_dir}/README.md")

            # Step 4: Push
            print(f"\n[4] Pushing to HuggingFace...")
            push_to_hf(out_dir, rname, args.dry_run)

        except Exception as e:
            print(f"\n  [ERROR] {model_id}: {e}")
            row["error"] = str(e)

        summary.append(row)

    # Print summary table
    print(f"\n\n{'='*80}")
    print(f"{'MODEL':<50} {'F16':>6} {'Q8':>6} {'Q4':>6}  STATUS")
    print('-'*80)
    for r in summary:
        err  = r.get('error') or ''
        cer_f16  = f"{r['cer_f16']:.2%}"  if r['cer_f16']  is not None else 'skip'
        cer_q8_0 = f"{r['cer_q8_0']:.2%}" if r['cer_q8_0'] is not None else 'skip'
        cer_q4_0 = f"{r['cer_q4_0']:.2%}" if r['cer_q4_0'] is not None else 'skip'
        status   = 'ERROR' if err else 'OK'
        short_id = r['model'].split('/')[-1][:47]
        print(f"{short_id:<50} {cer_f16:>6} {cer_q8_0:>6} {cer_q4_0:>6}  {status}")
        if err: print(f"  Error: {err}")
    print('='*80)

    failed = sum(1 for r in summary if r.get('error'))
    print(f"\nDone: {len(summary)-failed}/{len(summary)} models succeeded.")
    return 0 if not failed else 1

if __name__ == '__main__':
    sys.exit(main())
