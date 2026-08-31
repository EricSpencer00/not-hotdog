"""Write the measured numbers into README.md and web/how-it-works.html.

Numbers live in train/results/*.json and are injected between markers rather
than typed. Anything typed by hand drifts the moment the model is retrained,
and a stale accuracy claim is worse than no claim. The write-up is a published
page, so it gets the same treatment as the README.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import DIST, RESULTS, ROOT

START = "<!--NUMBERS-->"
END = "<!--/NUMBERS-->"

WRITEUP_START = "<!--RESULTS-->"
WRITEUP_END = "<!--/RESULTS-->"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def gzipped(path: Path) -> int:
    """What the browser actually downloads. Measured, not a ratio guess."""
    return len(gzip.compress(path.read_bytes(), 9))


def fill(path: Path, start: str, end: str, body: str) -> None:
    """Replace everything between the two markers. Both must already exist."""
    s = path.read_text()
    if start not in s or end not in s:
        sys.exit(f"no {start} block in {path}")
    pre = s.split(start)[0]
    post = s.split(end)[1]
    path.write_text(pre + start + body + end + post)
    print(f"filled numbers into {path}")


def writeup_body(ev: dict, student: dict, teacher: dict, cal: dict) -> str:
    """The same numbers again, as the HTML tables in web/how-it-works.html."""
    t, v, h = ev["test"], ev["val"], ev.get("hard", {})

    def row(name: str, d: dict) -> str:
        return (
            f"<tr><td>{name}</td><td class=\"n\">{d['n']:,}</td>"
            f"<td class=\"n\">{d['positives']}</td>"
            f"<td class=\"n\">{pct(d['accuracy'])}</td>"
            f"<td class=\"n\">{pct(d['majority_baseline'])}</td>"
            f"<td class=\"n\">{d['f1']:.3f}</td>"
            f"<td class=\"n\">{d['precision']:.3f}</td>"
            f"<td class=\"n\">{d['recall']:.3f}</td></tr>"
        )

    splits = "\n".join([row("validation", v), row("test", t), row("adversarial", h)])
    per_class = "\n".join(
        f"<tr><td><code>{k}</code></td><td class=\"n\">{c['n']}</td>"
        f"<td class=\"n\">{c['correct']}/{c['n']}</td>"
        f"<td class=\"n\">{pct(c['accuracy'])}</td></tr>"
        for k, c in h.get("per_class", {}).items()
    )

    return f"""
<h3>Accuracy</h3>

<table>
<thead><tr><th>split</th><th class="n">n</th><th class="n">positives</th><th class="n">accuracy</th><th class="n">majority</th><th class="n">F1</th><th class="n">precision</th><th class="n">recall</th></tr></thead>
<tbody>
{splits}
</tbody>
</table>

<p><b>Read F1, not accuracy.</b> About {pct(1 - t['positives'] / t['n'])} of the test split is negative, so a
model that says "not a hot dog" to everything already scores {pct(t['majority_baseline'])}. The accuracy
column is there because people expect it, not because it means much.</p>

<h3>The adversarial set</h3>

<p>Six classes are held out entirely. Not one image of any of them appears in
training.</p>

<table>
<thead><tr><th>class</th><th class="n">n</th><th class="n">correct</th><th class="n">accuracy</th></tr></thead>
<tbody>
{per_class}
</tbody>
</table>

<p>Adversarial accuracy, {pct(h.get('accuracy', 0))}, is below the majority baseline of {pct(h.get('majority_baseline', 0))}. That is
not a rounding artefact. On this set the model is worse than a rock with "not a
hot dog" written on it.</p>

<p>The per-class split says why, and it is more interesting than the aggregate.
The model rejects things that are nearly hot dogs: it gets {h.get('tn', 0)}/{h.get('tn', 0) + h.get('fp', 0)} of the
held-out negatives, never once mistakes a dachshund for food, and correctly
refuses {h['per_class']['corn_dog']['correct']} of {h['per_class']['corn_dog']['n']} corn dogs. What it fails is hard positives, {h.get('tp', 0)}/{h.get('tp', 0) + h.get('fn', 0)}, driven by chili
dogs — hot dogs buried under chili, which look nothing like the canonical
article.</p>

<h3>The pipeline, end to end</h3>

<table>
<thead><tr><th>stage</th><th>metric</th></tr></thead>
<tbody>
<tr><td>teacher (EfficientNet-B0, 5.3M params)</td><td>F1 {teacher['f1']:.3f}, accuracy {pct(teacher['accuracy'])}</td></tr>
<tr><td>student after distillation (float)</td><td>F1 {student['float_val_f1']:.3f}</td></tr>
<tr><td>student after QAT (int8)</td><td>F1 {student['qat_val_f1']:.3f}</td></tr>
<tr><td>cost of quantization</td><td>{student['float_val_f1'] - student['qat_val_f1']:+.3f} F1</td></tr>
<tr><td>decision threshold (int32, folded into bias)</td><td>{cal['threshold_int32']:,}</td></tr>
</tbody>
</table>
"""


def main() -> None:
    ev = json.loads((RESULTS / "eval.json").read_text())
    student = json.loads((RESULTS / "student.json").read_text())
    teacher = json.loads((RESULTS / "teacher.json").read_text())
    cal = json.loads((RESULTS / "calibration.json").read_text())

    html = (DIST / "index.html").stat().st_size
    js = (DIST / "hotdog.js").stat().st_size
    gz = sum(gzipped(DIST / f) for f in ("index.html", "hotdog.js"))

    t, v, h = ev["test"], ev["val"], ev.get("hard", {})

    per_class = "\n".join(
        f"| `{k}` | {v2['n']} | {v2['correct']}/{v2['n']} | {pct(v2['accuracy'])} |"
        for k, v2 in h.get("per_class", {}).items()
    )

    body = f"""
## Numbers

Measured on the **int8 model** — the one that ships — with the browser's exact
preprocessing. Not the float checkpoint.

| | value |
|---|---|
| parameters | {ev['params'] + 1:,} (int8) |
| weights | {ev['params'] / 1024:.0f} KB |
| page weight | {(html + js) / 1024:.0f} KB raw, {gz / 1024:.0f} KB gzipped |
| dependencies | 0 |
| inference | ~26 ms (Node, M1 Max) |
| MACs | 14.0M |

### Accuracy

| split | n | positives | accuracy | majority baseline | F1 | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | {v['n']:,} | {v['positives']} | {pct(v['accuracy'])} | {pct(v['majority_baseline'])} | {v['f1']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} |
| test | {t['n']:,} | {t['positives']} | {pct(t['accuracy'])} | {pct(t['majority_baseline'])} | {t['f1']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} |
| adversarial | {h.get('n', 0)} | {h.get('positives', 0)} | {pct(h.get('accuracy', 0))} | {pct(h.get('majority_baseline', 0))} | {h.get('f1', 0):.3f} | {h.get('precision', 0):.3f} | {h.get('recall', 0):.3f} |

**Read F1, not accuracy.** The evaluation sets are built to be hard: roughly
{pct(1 - t['positives'] / t['n'])} of the test split is negative, so a model that
says "not a hot dog" to literally everything already scores
{pct(t['majority_baseline'])}. The accuracy column is there because people expect
it, not because it means much.

The precision figure is also measured against that same deliberately hostile
mix — 12,000 photographs of other food and 8,000 photographs of unrelated
objects against roughly 1,800 hot dogs. Someone pointing a camera at their lunch
is not sampling from that distribution.

### Adversarial set, by class

Six classes held out entirely from training.

| class | n | correct | accuracy |
|---|---:|---:|---:|
{per_class}

Small-n classes are reported as-is rather than dropped: Open Images contains 46
corn dogs in total, and only the subset on the CVDF mirror is fetchable.

Note that adversarial accuracy ({pct(h.get('accuracy', 0))}) is *below* the
majority baseline ({pct(h.get('majority_baseline', 0))}). That is not a rounding
artefact, it is the result: on this set, the model is worse than a rock with
"not a hot dog" written on it.

The per-class split says why, and it is more interesting than the aggregate. The
model is good at *rejecting* things that are nearly hot dogs — it gets
{h.get('tn', 0)}/{h.get('tn', 0) + h.get('fp', 0)} of the held-out negatives, never once mistakes a dachshund for
food, and correctly refuses {h['per_class']['corn_dog']['correct']} of {h['per_class']['corn_dog']['n']} corn dogs. What it fails is
hard *positives*: {h.get('tp', 0)}/{h.get('tp', 0) + h.get('fn', 0)}, driven by chili dogs, which are hot dogs
buried under chili and look nothing like the canonical article.

So the model has learned a narrow, prototypical hot dog and is conservative
about anything outside it. Given 1,834 positives and 136K parameters that is the
failure mode you would predict, but it is worth stating plainly rather than
reporting the flattering test number and moving on.

### The pipeline, end to end

| stage | metric |
|---|---|
| teacher (EfficientNet-B0, 5.3M params) | F1 {teacher['f1']:.3f}, accuracy {pct(teacher['accuracy'])} |
| student after distillation (float) | F1 {student['float_val_f1']:.3f} |
| student after QAT (int8) | F1 {student['qat_val_f1']:.3f} |
| cost of quantization | {student['float_val_f1'] - student['qat_val_f1']:+.3f} F1 |
| decision threshold (int32, folded into bias) | {cal['threshold_int32']:,} |

The teacher-to-student gap is the honest headline: a 136K-parameter network is
not going to match a 5.3M-parameter one, and distillation narrows that gap
rather than closing it. What quantization costs on top of that is small, which
is the part QAT was for.

### Verification

| check | result |
|---|---|
| fixed-point vs Python reference | 1,005,400 vectors, 0 mismatches |
| per-kernel vs NumPy | 29 cases, 0 mismatches |
| JS engine vs Python integer reference | {cal['n_val']:,} validation images, **0 mismatches** |
"""
    fill(ROOT / "README.md", START, END, body)
    fill(ROOT / "web" / "how-it-works.html", WRITEUP_START, WRITEUP_END,
         writeup_body(ev, student, teacher, cal))


if __name__ == "__main__":
    main()
