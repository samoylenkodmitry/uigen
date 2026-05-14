# Generate Demo Music

Do not vendor ACE-Step into Cranamp. Install and run ACE-Step 1.5 outside this repository, then copy only reviewed raw exports into Cranamp for conversion.

References:

- ACE-Step 1.5 project: https://github.com/ace-step/ACE-Step-1.5
- ACE-Step 1.5 model: https://huggingface.co/ACE-Step/Ace-Step1.5
- ACE-Step 1.5 install docs: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INSTALL.md

## Arch/Linux NVIDIA Checklist

- Confirm the NVIDIA driver is working with `nvidia-smi`.
- Confirm Python 3.11 or 3.12 is available for ACE-Step.
- Install `uv` using the ACE-Step install guide.
- Install `ffmpeg` with `libmp3lame` support for final `.mp3` conversion.
- Keep the ACE-Step clone, virtual environment, model cache, and generated experiments outside this repo.

## Workflow

1. Clone and install ACE-Step 1.5 outside Cranamp by following the official install docs.
2. Start the ACE-Step UI or API from that external checkout.
3. Generate text-to-music tracks using `assets/demo-music/prompts/cranamp-demo-prompts.md`.
4. Use no reference audio, no vocals, no artist references, and no third-party samples.
5. Export raw audio as WAV or FLAC when possible.
6. Copy reviewed raw files into `assets/demo-music/generated/raw/`.
7. Name the raw files so the output stem matches the intended final filename, for example `cranamp-demo-01-retro-tracker.wav`.
8. Run:

```bash
scripts/convert_demo_music.sh
```

Use `--force` only when intentionally replacing existing generated `.mp3` files:

```bash
scripts/convert_demo_music.sh --force
```

9. Fill in `assets/demo-music/GENERATION.md` with generation date, seed, raw source filename, post-processing notes, and review notes.
10. Manually listen to each final `.mp3` before committing it.
11. Commit only selected final `.mp3` files after review.

## Conversion Fallback

The preferred output is MP3 through ffmpeg's `libmp3lame` encoder. If your ffmpeg build does not include `libmp3lame`, install a full ffmpeg build with LAME/MP3 support if possible.

If that is not practical, use another MP3-capable encoder only after documenting the exact command and codec in `assets/demo-music/GENERATION.md`. Do not silently mix codecs or commit unreviewed output.
