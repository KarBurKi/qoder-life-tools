# 实测：把 dsh-recall 的 mem_source（顺出处回原文）在 Qoder 日志上跑通
# 用法：python -X utf8 source_pull.py <jsonl路径> <查询> [k]
import json
import math
import re
import sys
import unicodedata

SECRETS = re.compile(r"(github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+|gho_[A-Za-z0-9]+|token[\"'\s:=]+[A-Za-z0-9_\-]{16,})", re.I)
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
OTHER_INJECT = re.compile(r"<(?:function_results|output|task-notification)>.*?</(?:function_results|output|task-notification)>", re.S)


def sanitize(text):
    """注入碎片不是谁说的话；整段是注入的直接丢，混在正文里的抠掉。最后脱敏。"""
    t = text.strip()
    if t.startswith("<system-reminder>"):
        return ""
    t = REMINDER.sub(" ", t)
    t = OTHER_INJECT.sub(" ", t)
    t = SECRETS.sub("[已脱敏]", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def tokens_of(text):
    s = text.lower()
    out = set()
    for m in re.finditer(r"[a-z][a-z0-9_./+-]{1,30}", s):
        out.add(m.group(0))
    han = "".join(re.findall(r"[\u3400-\u9fff]+", s))
    for i in range(len(han) - 1):
        out.add(han[i:i + 2])
    if len(han) == 1:
        out.add(han)
    return out


def stamp_of(row):
    for key in ("time", "timestamp", "ts", "created_at", "createdAt"):
        v = row.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return v / 1000 if v > 1e11 else v
        if isinstance(v, str) and v:
            return v
    return ""


def who(row):
    return "主人" if row.get("type") == "user" else "我"


def texts_of(row):
    """只取说出口的 text 块；thinking 是内心，tool_use/tool_result 不算原文。"""
    msg = row.get("message") if isinstance(row.get("message"), dict) else row
    content = msg.get("content")
    blocks = []
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        blocks = [b for b in content if isinstance(b, dict)]
    for b in blocks:
        if b.get("type") != "text":
            continue
        t = sanitize(b.get("text") or "")
        if len(t) >= 4:
            yield t


def collect(path):
    segs = []
    kinds = {}
    skipped_meta = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            kinds[row.get("type", "?")] = kinds.get(row.get("type", "?"), 0) + 1
            if row.get("type") not in ("user", "assistant"):
                continue
            if row.get("isMeta"):
                skipped_meta += 1
                continue
            for t in texts_of(row):
                segs.append({"who": who(row), "t": stamp_of(row), "x": t})
    return segs, kinds, skipped_meta


def rank(segs, query, k, min_hits=2):
    """会话段落不是记忆卡片：这里必须改。
    照搬卡片那套 BM25 长度归一化，实测会把「public就行」这种短句顶到前三——
    短文档 norm 小、得分虚高，真正的原文（千字段）被压下去。
    所以改成：IDF 加权交集，除以 sqrt(段长) 做软归一；再要求至少命中 min_hits 个查询词；
    查询里 4 字以上的连续串整串出现时重奖（同义改写靠这个兜）。
    """
    q = tokens_of(query)
    docs = [{"s": seg, "tok": tokens_of(seg["x"]), "n": max(1, len(seg["x"]))} for seg in segs]
    if not q:
        return docs[-k:][::-1]
    n = len(docs) or 1
    df = {}
    for d in docs:
        for tok in d["tok"]:
            df[tok] = df.get(tok, 0) + 1
    phrases = [p for p in re.split(r"\s+", query.strip()) if len(p) >= 4]
    scored = []
    for d in docs:
        hits = [tok for tok in q if tok in d["tok"]]
        if len(hits) < min_hits:
            continue
        sc = sum(math.log(1 + (n - df[tok] + 0.5) / (df[tok] + 0.5)) for tok in hits)
        sc /= math.sqrt(d["n"] / 40.0)
        sc *= 0.6 + 0.4 * (len(hits) / len(q))
        for p in phrases:
            if p.lower() in d["s"]["x"].lower():
                sc += 3.0
        scored.append((sc, d["s"]))
    scored.sort(key=lambda x: -x[0])
    return [seg for _, seg in scored[:k]]


def mask_check(path):
    """只统计，不打印正文：验证脱敏确实拦得住凭据。"""
    raw = open(path, encoding="utf-8", errors="replace").read()
    by_block = {}
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        msg = row.get("message") if isinstance(row.get("message"), dict) else row
        content = msg.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else [
            b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
        for b in blocks:
            blob = json.dumps(b, ensure_ascii=False)
            c = len(SECRETS.findall(blob))
            if c:
                key = "%s/%s" % (row.get("type"), b.get("type"))
                by_block[key] = by_block.get(key, 0) + c
    segs, _, _ = collect(path)
    after = sum(len(SECRETS.findall(s["x"])) for s in segs)
    leaked = sum(1 for s in segs if "[已脱敏]" in s["x"])
    print("文件原始凭据命中：%d 处，分布（行类型/块类型）：%s"
          % (len(SECRETS.findall(raw)), by_block or "无"))
    print("提取出的原文段落里仍含凭据：%d 处" % after)
    print("被替换过的段落数：%d（这些段只以 [已脱敏] 出现，不打印原值）" % leaked)
    print("结论：", "拦住了" if after == 0 else "没拦住，不能上")


def find_session(prefix):
    """跨会话定位：scope 不是目录名而是 projects/<工作区slug>/<sid>.jsonl。"""
    import glob
    import os
    root = os.path.expanduser("~/.qoder/projects")
    return [p for p in glob.glob(os.path.join(root, "*", "*" + prefix + "*.jsonl"))]


def main():
    path = sys.argv[1]
    if "--find" in sys.argv:
        for p in find_session(path):
            print(p)
        return
    if "--mask-check" in sys.argv:
        mask_check(path)
        return
    query = sys.argv[2] if len(sys.argv) > 2 else ""
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    segs, kinds, meta = collect(path)
    print("行类型分布：", kinds)
    print("可当原文的段落：%d 段（因 isMeta 跳过 %d 行）" % (len(segs), meta))
    lens = sorted(len(s["x"]) for s in segs)
    if lens:
        print("段长：最短 %d / 中位 %d / 最长 %d" % (lens[0], lens[len(lens) // 2], lens[-1]))
    print("查询：", query)
    print()
    got = rank(segs, query, k, 2)
    label = ""
    if not got:
        got = rank(segs, query, k, 1)
        label = "降级：严格档（至少两个词）一个都没中，放宽到命中一个词再看——多半是说法和日志里不一样\n"
    if not got:
        got = segs[-k:][::-1]
        label = "兜底：一次都没命中，只给结尾几段。这是「那次会话最后聊了什么」，不是相关结果，别当证据用\n"
    if label:
        print(label)
    for i, seg in enumerate(got, 1):
        body = seg["x"]
        if len(body) > 500:
            body = body[:500] + "…"
        print("%d. %s  %s" % (i, seg["who"], seg["t"]))
        print(body)
        print("-" * 60)


if __name__ == "__main__":
    main()
