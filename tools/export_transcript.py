"""Export a Claude Code session transcript (.jsonl) to a searchable Markdown file.

Keeps the user's messages and the assistant's text replies (tool calls are listed as one-line
stubs, tool outputs and images are dropped). Credentials are redacted: Kaggle tokens (KGAT_...),
generic API keys and bearer tokens never reach the output.

    python tools/export_transcript.py <session.jsonl> docs/conversation_log.md
"""

import json
import re
import sys

REDACT = [
    (re.compile(r"KGAT_[0-9a-fA-F]{16,}"), "KGAT_[REDACTED]"),
    (re.compile(r"\b(hf|ghp|gho|sk|xox[bp])[-_][A-Za-z0-9_\-]{16,}"), "[REDACTED_KEY]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{20,}"), r"\1[REDACTED]"),
]


def clean(text: str) -> str:
    for rx, rep in REDACT:
        text = rx.sub(rep, text)
    return text


def blocks(msg):
    c = msg.get("content")
    if isinstance(c, str):
        yield "text", c
        return
    for b in c or []:
        t = b.get("type")
        if t == "text":
            yield "text", b.get("text", "")
        elif t == "tool_use":
            inp = b.get("input", {})
            desc = inp.get("description") or inp.get("command") or inp.get("file_path") or ""
            yield "tool", f"{b.get('name')}: {str(desc)[:140]}"


def main(src, dst):
    out = ["# Nhật ký trao đổi (xuất tự động, đã che token)\n",
           "Nguồn: transcript phiên Claude Code. Chỉ giữ tin nhắn của người dùng và câu trả lời của trợ lý; "
           "lệnh gọi công cụ chỉ còn một dòng, kết quả công cụ và ảnh bị bỏ. Tìm nhanh bằng `grep -n`.\n"]
    n_user = 0
    with open(src) as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = d.get("type")
            if t not in ("user", "assistant"):
                continue
            msg = d.get("message", {})
            ts = (d.get("timestamp") or "")[:16].replace("T", " ")
            parts = []
            for kind, text in blocks(msg):
                text = clean(text).strip()
                if not text:
                    continue
                if kind == "tool":
                    parts.append(f"> _tool_ {text}")
                elif t == "user" and (text.startswith("<system-reminder>") or text.startswith("<local-command")
                                      or text.startswith("<artifact-view-context")):
                    continue
                else:
                    parts.append(text)
            if t == "user" and not any(not p.startswith("> _tool_") for p in parts):
                continue  # tool results only
            if not parts:
                continue
            if t == "user":
                n_user += 1
            out.append(f"\n---\n### {'NGƯỜI DÙNG' if t == 'user' else 'TRỢ LÝ'} · {ts}\n")
            out.append("\n\n".join(parts) + "\n")
    text = "\n".join(out)
    assert "KGAT_" not in text.replace("KGAT_[REDACTED]", ""), "token leak"
    with open(dst, "w") as f:
        f.write(text)
    print(f"wrote {dst}: {n_user} user messages, {len(text) / 1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
