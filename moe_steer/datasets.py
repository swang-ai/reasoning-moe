from __future__ import annotations

import re

import numpy as np

_REGISTRY = {}


def domain(name):
    def deco(cls):
        _REGISTRY[name] = cls()
        return cls
    return deco


def get_domain(name):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown domain '{name}'. Known: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_domains():
    return list(_REGISTRY)


def _sample(items, n, seed):
    if n is None or n >= len(items):
        return list(items)
    rng = np.random.default_rng(seed)
    return [items[i] for i in rng.choice(len(items), n, replace=False)]


def _last_boxed_or_number(text):
    seg = text.split("assistantfinal")[-1] if "assistantfinal" in text else text
    m = re.findall(r"\\boxed\{([^}]*)\}", seg)
    if m:
        return m[-1].replace(",", "").replace("$", "").strip()
    nums = re.findall(r"-?\d[\d,]*\.?\d*", seg)
    return nums[-1].replace(",", "") if nums else None


def _find_split_domain(name):
    @domain(f"{name}_find")
    class _Find:
        _file = f"{name}_find.json"

        def load(self, n=100, seed=0):
            import json
            from pathlib import Path
            p = Path(__file__).resolve().parents[1] / "data" / self._file
            return _sample(json.loads(p.read_text()), n, seed)
    return _Find


for _b in ("gpqa",):
    _find_split_domain(_b)


@domain("mathmix_find")
class MathMixFind:

    def load(self, n=300, seed=0):
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "data" / "mathmix_find.json"
        return _sample(json.loads(p.read_text()), n, seed)


@domain("mathmix_aime_find")
class MathMixAimeFind:

    def load(self, n=330, seed=0):
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "data" / "mathmix_aime_find.json"
        return _sample(json.loads(p.read_text()), n, seed)


@domain("gsm8k")
class GSM8K:

    def load(self, n=200, seed=0):
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        items = [{"problem": r["question"],
                  "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
                 for r in ds]
        return _sample(items, n, seed)

    def grade(self, text, gold):
        a = _last_boxed_or_number(text)
        if a is None:
            return False
        try:
            return abs(float(a) - float(str(gold).replace(",", ""))) < 1e-6
        except ValueError:
            return a == str(gold).strip()


@domain("math500")
class MATH500:

    def load(self, n=200, seed=0, level=None):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"problem": r["problem"], "gold": r["answer"], "level": r.get("level")}
                 for r in ds]
        if level is not None:
            items = [p for p in items if p.get("level") in level]
        return _sample(items, n, seed)

    def grade(self, text, gold):
        from math_verify import parse, verify
        seg = text.split("assistantfinal")[-1] if "assistantfinal" in text else text
        m = re.findall(r"\\boxed\{(.+?)\}", seg)
        pred = m[-1] if m else _last_boxed_or_number(text)
        if pred is None:
            return False
        try:
            return bool(verify(parse(str(gold)), parse(pred)))
        except Exception:
            return str(pred).strip() == str(gold).strip()


@domain("numinamath")
class NuminaMath:

    def load(self, n=400, seed=0, hf_id="AI-MO/NuminaMath-CoT"):
        from datasets import load_dataset
        ds = load_dataset(hf_id, split="train")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
        items = []
        for i in idx:
            r = ds[int(i)]
            sol = r.get("solution", "") or ""
            m = re.findall(r"\\boxed\{(.+?)\}", sol)
            items.append({"problem": r["problem"], "gold": (m[-1] if m else None)})
        return items

    def grade(self, text, gold):
        if gold is None:
            return False
        seg = text.split("assistantfinal")[-1] if "assistantfinal" in text else text
        m = re.findall(r"\\boxed\{(.+?)\}", seg)
        pred = m[-1] if m else _last_boxed_or_number(text)
        if pred is None:
            return False
        try:
            from math_verify import parse, verify
            return bool(verify(parse(str(gold)), parse(pred)))
        except Exception:
            return str(pred).strip() == str(gold).strip()


@domain("gpqa")
class GPQA:

    _INSTR = ("\n\nThink step by step, then give ONLY the letter (A, B, C, or D) "
              "of the correct option, as \\boxed{X}.")

    def load(self, n=198, seed=0, domain=None):
        from datasets import load_dataset
        ds = load_dataset("fingertap/GPQA-diamond", split="test")
        items = [{"problem": r["question"] + self._INSTR,
                  "gold": str(r["answer"]).strip().upper()} for r in ds]
        return _sample(items, n, seed)

    def grade(self, text, gold):
        seg = text.split("assistantfinal")[-1] if "assistantfinal" in text else text
        m = re.findall(r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}", seg)
        if not m:
            m = re.findall(r"(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?([A-Da-d])\)?\b",
                           seg, re.I)
        if not m:
            m = re.findall(r"\b([A-D])\b", seg)
        if not m:
            return False
        return m[-1].strip().upper() == str(gold).strip().upper()
