"""Create .env.prod (strong random secrets) from .env.prod.example. Refuses to overwrite unless --force.

  python deploy/gen_secrets.py --domain support.example.com --email ops@example.com
"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="localhost")
    ap.add_argument("--email", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = ROOT / ".env.prod"
    if out.exists() and not args.force:
        print(f"{out} exists; use --force to regenerate (this rotates every secret).", file=sys.stderr)
        return 1
    lines, keys = [], {}
    for line in (ROOT / ".env.prod.example").read_text(encoding="utf-8").splitlines():
        if line.startswith("SITE_ADDRESS="):
            line = f"SITE_ADDRESS={args.domain}"
        elif line.startswith("TLS_MODE="):
            if args.domain != "localhost" and not args.email:
                sys.exit("--email is required for a public domain (Let's Encrypt contact)")
            line = f"TLS_MODE={args.email or 'internal'}"
        elif "__GENERATE__" in line:
            name = line.split("=", 1)[0]
            if name == "API_KEYS":
                keys = {"agents": secrets.token_urlsafe(32), "service": secrets.token_urlsafe(32)}
                value = ",".join(keys.values())
            else:  # URL-safe and without characters that need escaping in connection strings
                value = secrets.token_urlsafe(24)
            line = f"{name}={value}"
        lines.append(line)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sec = ROOT / "secrets"
    sec.mkdir(exist_ok=True)
    token = next(l.split("=", 1)[1] for l in lines if l.startswith("METRICS_TOKEN="))
    (sec / "metrics_token").write_text(token, encoding="utf-8")
    try:
        out.chmod(0o600)
        (sec / "metrics_token").chmod(0o600)
    except OSError:
        pass
    print(f"wrote {out} and secrets/metrics_token")
    print("API keys (store them in your password manager; they are not shown again):")
    for name, k in keys.items():
        print(f"  {name}: {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
