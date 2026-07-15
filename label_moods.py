#!/usr/bin/env python
"""
label_moods.py

Labels contiguous Arabic poetry verse-batches along FOUR independent axes
using BGE-M3 embeddings + on-the-fly Z-SCORE calibration to eliminate hub-label bias.

Adds, for each axis <axis> in {mood, genre, energy, aesthetic}:
    - <axis>_tags        : list[str]        Arabic tags for that axis
    - <axis>_scores      : dict[str, float] raw cosine similarities
    - <axis>_scores_z    : dict[str, float] on-the-fly standardized Z-scores
    - <axis>_confidence  : float             top-1 similarity score (Z-score)
    - <axis>_top2_gap    : float             top1_z - top2_z (Z-score gap)
    - <axis>_low_confidence : bool          True if gap is under threshold
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Taxonomies
# ---------------------------------------------------------------------------
MOOD_TAXONOMY = {
    "فرح": {
        "embed": "أبيات فرح وسرور واحتفال بمناسبة سعيدة",
        "suno": "joyful, uplifting",
    },
    "حزن": {
        "embed": "أبيات حزن وأسى وكآبة دون ان تكون رثاء لشخص متوفى بعينه",
        "suno": "sad, sorrowful",
    },
    "غضب": {
        "embed": "أبيات غضب وسخط وثورة على ظلم او اهانة",
        "suno": "angry, defiant",
    },
    "تشاؤم": {
        "embed": "أبيات تشاؤم ويأس من المستقبل",
        "suno": "bleak, hopeless",
    },
    "تفاؤل وأمل": {
        "embed": "أبيات تفاؤل وأمل بالمستقبل وتجدد",
        "suno": "hopeful, optimistic",
    },
    "حنين وشوق": {
        "embed": "أبيات حنين وشوق الى الاحبة أو الوطن أو الماضي",
        "suno": "nostalgic, longing",
    },
    "وحدة": {
        "embed": "أبيات وحدة وعزلة واغتراب عن الناس",
        "suno": "lonely, isolated",
    },
    "شكوى": {
        "embed": "أبيات شكوى وتذمر من الزمان أو الظلم أو سوء الحظ",
        "suno": "bitter, complaining",
    },
    "عتاب": {
        "embed": "أبيات عتاب ولوم موجهة لصديق أو حبيب بسبب جفاء أو خيانة",
        "suno": "reproachful, wounded",
    },
    "تأمل": {
        "embed": "أبيات تأمل هادئ في معنى الحياة أو الكون",
        "suno": "contemplative, reflective",
    },
}

GENRE_TAXONOMY = {
    "مدح": {
        "embed": "قصيدة مدح وثناء تمجد شخصا وتصف صفاته الحميدة وكرمه وشجاعته",
        "suno": "praise, tribute",
    },
    "هجاء": {
        "embed": "قصيدة هجاء وسخرية وذم تنتقد شخصا وتصفه بصفات سيئة",
        "suno": "satirical, mocking",
    },
    "رثاء": {
        "embed": "قصيدة رثاء وحزن على فقد شخص متوفى وتأبين له",
        "suno": "elegy, mourning",
    },
    "غزل": {
        "embed": "قصيدة غزل وحب ووصف جمال المحبوب وشوق عاطفي وحسي",
        "suno": "romantic, love song",
    },
    "فخر": {
        "embed": "قصيدة فخر واعتزاز بالنفس والقبيلة والانتصارات والانجازات",
        "suno": "boastful, triumphant",
    },
    "حكمة": {
        "embed": "أبيات حكمة وتأمل فلسفي في الحياة والموت والمصير",
        "suno": "wise, philosophical",
    },
    "زهد": {
        "embed": "أبيات زهد وتقوى ونصح بالابتعاد عن ملذات الدنيا والتقرب الى الله",
        "suno": "ascetic, devotional",
    },
    "وصف": {
        "embed": "أبيات وصف للطبيعة والاماكن والاشياء دون مضمون عاطفي قوي",
        "suno": "descriptive, scenic",
    },
    "خمريات": {
        "embed": "أبيات خمريات تصف الخمر والسكر والمجون واللهو",
        "suno": "hedonistic, revelrous",
    },
    "حماسة": {
        "embed": "أبيات حماسة وشجاعة في الحرب والقتال والبطولة",
        "suno": "heroic, martial",
    },
}

ENERGY_TAXONOMY = {
    "هادئ جدا": {
        "embed": "أبيات هادئة بطيئة الايقاع توحي بالسكون والراحة والاسترخاء التام",
        "suno": "calm, slow tempo, ambient",
    },
    "هادئ": {
        "embed": "أبيات هادئة متأملة معتدلة الايقاع قليلة التوتر",
        "suno": "mellow, relaxed, low energy",
    },
    "متوسط": {
        "embed": "أبيات معتدلة الحيوية متوسطة الايقاع والتوتر لا هي هادئة ولا صاخبة",
        "suno": "mid-tempo, moderate energy",
    },
    "نشيط": {
        "embed": "أبيات نشيطة حيوية متسارعة الايقاع تحمل طاقة وحركة",
        "suno": "upbeat, driving, energetic",
    },
    "شديد الحماس": {
        "embed": "أبيات شديدة الحماس والتوتر والانفعال سريعة قوية الوقع",
        "suno": "high-energy, intense, fast tempo",
    },
}

AESTHETIC_TAXONOMY = {
    "تراثي أصيل": {
        "embed": "اجواء تراثية اصيلة توحي بالعود والانشاد التقليدي والموروث الشعبي القديم",
        "suno": "traditional, heritage, oud, acoustic",
    },
    "ملحمي أوركسترالي": {
        "embed": "اجواء ملحمية ضخمة توحي بموسيقى اوركسترالية سينمائية وحشود ومعارك",
        "suno": "epic, orchestral, cinematic",
    },
    "صوفي روحاني": {
        "embed": "اجواء صوفية روحانية توحي بالانشاد الديني والتأمل والسمو الروحي",
        "suno": "mystical, spiritual, ethereal, ambient",
    },
    "عسكري حماسي": {
        "embed": "اجواء عسكرية حماسية توحي بطبول الحرب والنشيد الجماعي والزحف",
        "suno": "martial, anthem, percussive, war drums",
    },
    "رومانسي عاطفي": {
        "embed": "اجواء رومانسية عاطفية حميمة توحي بأغنية حب هادئة ودافئة",
        "suno": "romantic, tender, acoustic ballad",
    },
    "حزين كئيب": {
        "embed": "اجواء حزينة كئيبة توحي بموسيقى بطيئة على سلم صغير ونغمة معتمة",
        "suno": "melancholic, somber, minor key, slow",
    },
    "احتفالي شعبي": {
        "embed": "اجواء احتفالية شعبية توحي بموسيقى فرح جماعي راقص ومهرجاني",
        "suno": "festive, folk, upbeat, celebratory",
    },
}

AXES = {
    "mood": {
        "taxonomy": MOOD_TAXONOMY,
        "min_tags": 2,
        "max_tags": 4,
        "margin": 0.5,
        "low_margin_threshold": 0.3,
    },
    "genre": {
        "taxonomy": GENRE_TAXONOMY,
        "min_tags": 1,
        "max_tags": 3,
        "margin": 0.5,
        "low_margin_threshold": 0.3,
    },
    "energy": {
        "taxonomy": ENERGY_TAXONOMY,
        "min_tags": 1,
        "max_tags": 1,
        "margin": 0.5,
        "low_margin_threshold": 0.4,
    },
    "aesthetic": {
        "taxonomy": AESTHETIC_TAXONOMY,
        "min_tags": 1,
        "max_tags": 3,
        "margin": 0.5,
        "low_margin_threshold": 0.3,
    },
}

MODEL_NAME = "BAAI/bge-m3"


def build_batch_text(data):
    """Concatenate sadr+ajuz of every verse in a batch into one text block."""
    parts = []
    for verse in data:
        sadr = (verse.get("sadr") or "").strip()
        ajuz = (verse.get("ajuz") or "").strip()
        parts.append(f"{sadr} {ajuz}".strip())
    return " ".join(parts)


def select_tags(scores, margin=0.5, min_tags=2, max_tags=4):
    """
    scores: {label: z_score}
    Selects tags based on their standardised Z-score.
    Returns: tags list, top score, and raw z-score gap to the second prediction.
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_score = ranked[0][1]
    within_margin = [lbl for lbl, s in ranked if s >= top_score - margin]
    if len(within_margin) < min_tags:
        within_margin = [lbl for lbl, _ in ranked[:min_tags]]
    tags = within_margin[:max_tags]
    top2_gap = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else top_score
    return tags, top_score, top2_gap


def score_axis(batch_embs, model, taxonomy):
    """Encode one axis's taxonomy and score every batch embedding against it."""
    labels = list(taxonomy.keys())
    label_texts = [taxonomy[lbl]["embed"] for lbl in labels]
    label_embs = model.encode(
        label_texts, normalize_embeddings=True, convert_to_numpy=True
    )
    sims = batch_embs @ label_embs.T  # Cosine similarity matrix shape: (N, L)
    return labels, sims


def calibrate_scores(sims):
    """
    Compute column-wise (label-wise) mean and std deviation over the full batch.
    Converts raw cosines to Z-scores to eliminate structural bias.
    """
    means = np.mean(sims, axis=0)
    stds = np.std(sims, axis=0)
    stds = np.where(stds < 1e-9, 1.0, stds)  # Prevent division by zero
    z_scores = (sims - means) / stds
    return z_scores


def compute_embeddings_checkpointed(
    texts,
    model,
    checkpoint_dir,
    chunk_size,
    model_name,
    encode_batch_size,
    push_every,
    do_push,
    repo_dir,
):
    """
    Encode `texts` in row-order chunks, checkpointing each chunk to disk as
    it completes and (optionally) pushing checkpoints to git periodically.

    Safe to interrupt and re-run: any chunk already valid on disk is
    skipped, so a crash only ever costs the partially-encoded chunk that
    was in flight, not the whole run.
    """
    from checkpoint_utils import EmbeddingCheckpoint, git_checkpoint_push, chunk_bounds

    n_rows = len(texts)
    ckpt = EmbeddingCheckpoint(checkpoint_dir, n_rows, chunk_size, model_name)
    bounds = list(chunk_bounds(n_rows, chunk_size))
    total_chunks = len(bounds)

    already_done, _ = ckpt.progress(total_chunks)
    if already_done:
        print(
            f"      Resuming from checkpoint: {already_done}/{total_chunks} "
            f"chunks already done ({min(already_done * chunk_size, n_rows)}/{n_rows} rows)."
        )

    since_last_push = 0
    t0 = time.time()

    try:
        for chunk_idx, start, end in bounds:
            expected_rows = end - start

            if ckpt.is_done(chunk_idx, expected_rows):
                continue

            chunk_embs = model.encode(
                texts[start:end],
                batch_size=encode_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            ckpt.save_chunk(chunk_idx, chunk_embs)
            since_last_push += 1

            elapsed = time.time() - t0
            print(
                f"      chunk {chunk_idx + 1}/{total_chunks} "
                f"(rows {start}-{end}) encoded & checkpointed. "
                f"[{end}/{n_rows} rows, {elapsed:.0f}s elapsed]"
            )

            is_last_chunk = chunk_idx == total_chunks - 1
            if do_push and (since_last_push >= push_every or is_last_chunk):
                git_checkpoint_push(
                    repo_dir,
                    [str(ckpt.dir)],
                    message=(
                        f"checkpoint: embeddings through chunk {chunk_idx} "
                        f"(rows 0-{end}/{n_rows})"
                    ),
                )
                since_last_push = 0
    except (Exception, KeyboardInterrupt):
        print(
            "\n      Run interrupted or failed mid-encode. Everything checkpointed "
            "so far is safe on disk. Attempting a final push before raising ..."
        )
        if do_push:
            from checkpoint_utils import git_checkpoint_push as _push

            _push(
                repo_dir,
                [str(ckpt.dir)],
                message="checkpoint: partial progress (run interrupted)",
            )
        raise

    return ckpt.load_all(total_chunks)


def main():
    ap = argparse.ArgumentParser(
        description="Label Arabic verse batches on mood/genre/energy/aesthetic axes via embedding Z-Scores."
    )
    ap.add_argument("pkl_path", type=str, help="Path to input pkl dataframe")
    ap.add_argument("--out", type=str, default=None, help="Output pkl path")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu")
    ap.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    ap.add_argument("--margin", type=float, default=None, help="Override z-margin")
    ap.add_argument(
        "--low-margin-threshold",
        type=float,
        default=None,
        help="Override gap threshold",
    )
    ap.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for resumable embedding checkpoints (default: ./checkpoints)",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Rows per checkpoint chunk during embedding (default: 500)",
    )
    ap.add_argument(
        "--push-every",
        type=int,
        default=4,
        help="Git-push checkpoints every N completed chunks (default: 4)",
    )
    ap.add_argument(
        "--no-git-push",
        action="store_true",
        help="Save checkpoints locally but never git commit/push them",
    )
    ap.add_argument(
        "--repo-dir",
        type=str,
        default=".",
        help="Git repo root to commit/push from (default: current directory)",
    )
    args = ap.parse_args()

    pkl_path = Path(args.pkl_path)
    out_path = (
        Path(args.out)
        if args.out
        else pkl_path.with_name(pkl_path.stem + "_mood_labeled.pkl")
    )

    import pandas as pd

    print(f"[1/5] Loading dataframe from {pkl_path} ...")
    df = pd.read_pickle(pkl_path)
    print(f"      {len(df)} rows loaded.")

    print("[2/5] Building batch texts ...")
    df["_batch_text"] = df["DATA"].apply(build_batch_text)

    print(f"[3/5] Loading embedding model ({MODEL_NAME}) ...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit(
            "sentence-transformers not installed. Run: pip install sentence-transformers"
        )

    device = args.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      Using device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    print(
        f"[4/5] Encoding batch texts (chunk-size={args.chunk_size}, "
        f"checkpoint-dir={args.checkpoint_dir}, "
        f"git-push={'off' if args.no_git_push else f'every {args.push_every} chunks'}) ..."
    )
    batch_embs = compute_embeddings_checkpointed(
        texts=df["_batch_text"].tolist(),
        model=model,
        checkpoint_dir=args.checkpoint_dir,
        chunk_size=args.chunk_size,
        model_name=MODEL_NAME,
        encode_batch_size=args.batch_size,
        push_every=args.push_every,
        do_push=not args.no_git_push,
        repo_dir=args.repo_dir,
    )

    print("[5/5] Scoring + calibrating tags per axis ...")
    flagged_col = [[] for _ in range(len(df))]

    for axis_name, cfg in AXES.items():
        labels, sims = score_axis(batch_embs, model, cfg["taxonomy"])
        sims_z = calibrate_scores(sims)

        margin = args.margin if args.margin is not None else cfg["margin"]
        low_margin_threshold = (
            args.low_margin_threshold
            if args.low_margin_threshold is not None
            else cfg["low_margin_threshold"]
        )

        tags_col, scores_col, scores_z_col = [], [], []
        confidence_col, top2_gap_col, low_conf_col = [], [], []

        for i in range(len(sims)):
            row_raw_scores = {labels[j]: float(sims[i][j]) for j in range(len(labels))}
            row_z_scores = {labels[j]: float(sims_z[i][j]) for j in range(len(labels))}

            tags, confidence, top2_gap = select_tags(
                row_z_scores,
                margin=margin,
                min_tags=cfg["min_tags"],
                max_tags=cfg["max_tags"],
            )

            is_low_conf = top2_gap < low_margin_threshold

            tags_col.append(tags)
            scores_col.append(row_raw_scores)
            scores_z_col.append(row_z_scores)
            confidence_col.append(confidence)
            top2_gap_col.append(top2_gap)
            low_conf_col.append(is_low_conf)

            if is_low_conf:
                flagged_col[i].append(axis_name)

        df[f"{axis_name}_tags"] = tags_col
        df[f"{axis_name}_scores"] = scores_col
        df[f"{axis_name}_scores_z"] = scores_z_col
        df[f"{axis_name}_confidence"] = confidence_col
        df[f"{axis_name}_top2_gap"] = top2_gap_col
        df[f"{axis_name}_low_confidence"] = low_conf_col

    df["flagged_axes"] = flagged_col

    # Convenience column: Top mapped English Suno equivalents
    def top_label_suno(row):
        out = {}
        for axis_name, cfg in AXES.items():
            top_tags = row[f"{axis_name}_tags"]
            if top_tags:
                out[axis_name] = cfg["taxonomy"][top_tags[0]]["suno"]
        return out

    df["suno_tags"] = df.apply(top_label_suno, axis=1)
    df.drop(columns=["_batch_text"], inplace=True)

    df.to_pickle(out_path)
    print(f"Done. Saved -> {out_path}")

    if not args.no_git_push:
        from checkpoint_utils import git_checkpoint_push

        git_checkpoint_push(
            args.repo_dir,
            [str(out_path)],
            message=f"Final labeled output: {out_path.name} ({len(df)} rows)",
        )

    n_flagged = sum(1 for f in flagged_col if f)
    print(
        f"\n{n_flagged}/{len(df)} rows have at least one shaky (low-margin) axis pick."
    )
    for axis_name in AXES:
        n_axis_flagged = sum(1 for f in flagged_col if axis_name in f)
        print(f"  {axis_name}: {n_axis_flagged} flagged")

    sample_cols = [
        c
        for c in [
            "POET_NAME",
            "poem_no",
            "mood_tags",
            "genre_tags",
            "energy_tags",
            "aesthetic_tags",
            "flagged_axes",
        ]
        if c in df.columns
    ]
    print(df[sample_cols].sample(min(5, len(df))))


if __name__ == "__main__":
    main()
