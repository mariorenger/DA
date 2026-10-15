import numpy as np

from hgcd import chain as C
from hgcd import data as D
from hgcd.decoding import decide, jsd, prob_contrast, log_contrast

CHAIN = """Hop 1: Clarence's granddaughter is Emily. => Emily
Hop 2: Emily's brother is Michael. => Michael
Answer: Michael is Clarence's"""

BRIDGE = [C.Hop(1, "Who is the performer of Green?", "Steve Hillage"),
          C.Hop(2, "Who is the spouse of Steve Hillage?", "Miquette Giraudy")]


def test_parse_and_drg():
    hops = C.parse_chain(CHAIN)
    assert [h.result for h in hops] == ["Emily", "Michael"]
    assert C.drg_edges(hops) == [(0, 1)]
    assert C.descendants(hops, 0) == [1]
    assert C.relevant_hops(hops, target="Michael") == [0, 1]


def test_redundant_hop_not_relevant():
    hops = C.parse_chain(CHAIN)
    hops.insert(1, C.Hop(0, "Orrin's sister is Calla.", "Calla"))
    assert C.relevant_hops(hops, target="Michael") == [0, 2]
    assert C.descendants(hops, 1) == []


def test_do_total_vs_naive_bridge():
    naive = C.do_naive(BRIDGE, 0)
    total = C.do_total(BRIDGE, 0)
    # naive keeps the bridge entity and the answer downstream (leak -> direct effect ~ 0)
    assert "Steve Hillage" in naive[1].body and naive[1].result == "Miquette Giraudy"
    # total masks the bridge entity and the descendant's (answer) value
    assert "Steve Hillage" not in total[1].body and total[1].result == C.MASK
    assert total[0].body == C.MASK


def test_placebo_preserves_content():
    hops = C.parse_chain(CHAIN)
    p = C.placebo(hops, 0)
    assert p[0].body == "Emily is the granddaughter of Clarence." and p[0].result == "Emily"


def test_parse_tolerates_annotations():
    hops = C.parse_chain("Hop 1: A's father is B. => B (father)\nHop 2: junk\nHop 3: x => C.")
    assert [h.result for h in hops] == ["B", "C"]


def test_decoding_rules():
    p_cot = np.array([0.5, 0.4, 0.1])
    neg = np.array([0.6, 0.1, 0.3])      # candidate 0 is a shortcut (high without the hop)
    assert prob_contrast(p_cot, neg, alpha=1.0, beta=0.1) == 1
    assert log_contrast(p_cot, neg, alpha=1.0, beta=0.1) == 1
    assert prob_contrast(p_cot, neg, alpha=0.0, beta=0.1) == 0
    assert abs(jsd(p_cot, p_cot)) < 1e-9 and 0 < jsd(p_cot, neg) <= 1


def test_prop1_mixture_cancellation():
    # p = lam * p_chain + (1-lam) * p_short ; contrast in prob space cancels p_short exactly
    rng = np.random.default_rng(0)
    p_short = rng.dirichlet(np.ones(5))
    pc, pc_do = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
    lam = 0.3
    p, p_do = lam * pc + (1 - lam) * p_short, lam * pc_do + (1 - lam) * p_short
    assert np.allclose(p - p_do, lam * (pc - pc_do))
    # argmax of the alpha->inf prob contrast equals argmax of the chain-component effect
    assert np.argmax(p - p_do) == np.argmax(pc - pc_do)


def test_decide_all_methods():
    rec = dict(p_cot=np.array([0.5, 0.4, 0.1]), p_dir=np.array([0.3, 0.3, 0.4]),
               p_noctx=np.array([0.2, 0.2, 0.6]), p_tot=[np.array([0.6, 0.1, 0.3])],
               p_nai=[np.array([0.5, 0.4, 0.1])], p_pla=[np.array([0.5, 0.4, 0.1])], rel=[0])
    from hgcd.decoding import METHODS
    for m in METHODS:
        assert decide(rec, m) in (0, 1, 2)
    assert decide(rec, "hgcd_placebo") == decide(rec, "cot")


def test_clutrr_interventions_offline():
    ex = D.ClutrrExample(uid="u1", story="[Ann]'s son, [Bob], hugged his sister [Cat]. [Ann] smiled.",
                         names=["Ann", "Bob", "Cat"], genders={"Ann": "female", "Bob": "male",
                                                               "Cat": "female"},
                         source="Ann", target="Cat", answer="daughter",
                         gold_hops=[D.GoldHop("Ann", "son", "Bob"), D.GoldHop("Bob", "sister", "Cat")])
    r = D.rename(ex)
    assert r.answer == ex.answer and r.source != "Ann" and r.source in r.story_text
    assert D.reorder(ex).story != ex.story or len(D.split_sentences(ex.story)) == 1
    cf = D.counterfactual_story(ex)
    assert len(cf.split()) == len(ex.story_text.split()) and cf != ex.story_text
    pi = D.placebo_input(ex)
    assert all(n in pi.story_text for n in ex.names)
