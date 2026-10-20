"""Build the compressed (per-pixel, per-z-bin) randoms cache for one tracer.
See common.build_randoms_cache. Output: outputs/randoms_cache/<sample>_<T>.fits

    python build_randoms_cache.py --sample photo --tracer LRG1
"""
import argparse
import os
import time

from common import build_randoms_cache, cache_path, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True, choices=["spec", "photo"])
    p.add_argument("--tracer", required=True)
    args = p.parse_args()
    cfg = load_config()
    t0 = time.time()
    t = build_randoms_cache(cfg, args.sample, args.tracer, log=lambda s: print(s, flush=True))
    out = cache_path(cfg, args.sample, args.tracer)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t.write(out, overwrite=True)
    print(f"wrote {out}: {len(t)} rows, {int(t['NRAN'].sum()):,} randoms, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
