"""Few-shot prompts that elicit hop-structured chains."""

from __future__ import annotations

from typing import List

from .chain import Hop, render_chain

CLUTRR_SYSTEM = (
    "You solve family-relation puzzles. Walk from the first person to the second person "
    "one relation at a time, using only facts stated in the story. Write one line per hop "
    "in the form 'Hop i: X's <relation> is Y. => Y', then give the final answer.")

CLUTRR_SHOTS = [
    {
        "story": "Mabel took her son Rupert to the zoo. Rupert and his sister Agatha baked a cake "
                 "for their grandmother.",
        "src": "Mabel", "tgt": "Agatha",
        "hops": [("Mabel's son is Rupert.", "Rupert"), ("Rupert's sister is Agatha.", "Agatha")],
        "ans": "daughter",
    },
    {
        "story": "Clement went fishing with his father Bertram. Bertram called his sister Winifred "
                 "to wish her a happy birthday. Winifred was proud of her daughter Odette.",
        "src": "Clement", "tgt": "Odette",
        "hops": [("Clement's father is Bertram.", "Bertram"),
                 ("Bertram's sister is Winifred.", "Winifred"),
                 ("Winifred's daughter is Odette.", "Odette")],
        "ans": "cousin",
    },
]
# The second shot uses a relation outside the CLUTRR label set on purpose: it teaches the
# format without leaking label statistics. Its answer is never a scoring candidate.

MUSIQUE_SYSTEM = (
    "You answer multi-hop questions using the given paragraphs. Break the question into "
    "single-hop sub-questions, answer each in one line of the form "
    "'Hop i: <sub-question> => <answer>', then give the final answer.")

MUSIQUE_SHOTS = [
    {
        "context": "[1] Thornfield Abbey: Thornfield Abbey is a novel written by Edwina Marsh.\n"
                   "[2] Edwina Marsh: Edwina Marsh was born in Galway in 1902.\n"
                   "[3] Galway Bay: Galway Bay is a large bay on the west coast of Ireland.",
        "question": "Where was the author of Thornfield Abbey born?",
        "hops": [("Who is the author of Thornfield Abbey?", "Edwina Marsh"),
                 ("Where was Edwina Marsh born?", "Galway")],
        "ans": "Galway",
    },
]


def _clutrr_user(story: str, src: str, tgt: str) -> str:
    return f"Story: {story}\nQuestion: How is {tgt} related to {src}?"


def _hops_text(pairs) -> str:
    return render_chain([Hop(i + 1, b, r) for i, (b, r) in enumerate(pairs)])


def clutrr_messages(story: str, src: str, tgt: str) -> List[dict]:
    msgs = [{"role": "system", "content": CLUTRR_SYSTEM}]
    for s in CLUTRR_SHOTS:
        msgs.append({"role": "user", "content": _clutrr_user(s["story"], s["src"], s["tgt"])})
        msgs.append({"role": "assistant", "content": _hops_text(s["hops"]) +
                     f"Answer: {s['tgt']} is {s['src']}'s {s['ans']}."})
    msgs.append({"role": "user", "content": _clutrr_user(story, src, tgt)})
    return msgs


def musique_messages(context: str, question: str) -> List[dict]:
    msgs = [{"role": "system", "content": MUSIQUE_SYSTEM}]
    for s in MUSIQUE_SHOTS:
        msgs.append({"role": "user", "content": f"Paragraphs:\n{s['context']}\nQuestion: {s['question']}"})
        msgs.append({"role": "assistant", "content": _hops_text(s["hops"]) + f"Answer: {s['ans']}"})
    msgs.append({"role": "user", "content": f"Paragraphs:\n{context}\nQuestion: {question}"})
    return msgs


def musique_question_only(question: str) -> List[dict]:
    """No-context prompt (for CAD's contrast distribution)."""
    return musique_messages("(no paragraphs given)", question)


def clutrr_question_only(src: str, tgt: str) -> List[dict]:
    return clutrr_messages("(no story given)", src, tgt)
