from collections import Counter

from mmft.data import INSTRUCTION, MCItem, format_prompt, load_v7w, shuffle_choices


def test_format_prompt():
    item = MCItem("1", "x.jpg", "Where is he sitting?", ("At a park.", "On a bench."), 1, "where")
    assert format_prompt(item) == f"Where is he sitting?\nA. At a park.\nB. On a bench.\n{INSTRUCTION}"
    with_ctx = MCItem("2", "x.jpg", "Which is north?", ("a", "b", "c"), 0, "social science", context="A map.")
    assert format_prompt(with_ctx).startswith("Context: A map.\nWhich is north?\nA. a\nB. b\nC. c\n")


def test_shuffle_is_deterministic_and_uniform():
    choices = ["right", "w1", "w2", "w3"]
    assert shuffle_choices("42", choices, 0, 0) == shuffle_choices("42", choices, 0, 0)
    positions = Counter(shuffle_choices(str(i), choices, 0, 0)[1] for i in range(8000))
    assert set(positions) == {0, 1, 2, 3}
    assert all(abs(c / 8000 - 0.25) < 0.02 for c in positions.values())
    for i in range(50):
        opts, ans = shuffle_choices(str(i), choices, 0, 7)
        assert opts[ans] == "right" and sorted(opts) == sorted(choices)


def test_load_v7w_splits_and_filtering(v7w_root):
    train = load_v7w(v7w_root, "train")
    assert len(train) == 6 * 3  # the duplicate-option item is dropped
    assert all(it.choices[it.answer].startswith("right") for it in train)
    assert {it.group for it in train} == {"what", "where", "why"}
    assert len(load_v7w(v7w_root, "val")) == 6 and len(load_v7w(v7w_root, "test")) == 9
    assert not {it.qid for it in train} & {it.qid for it in load_v7w(v7w_root, "test")}


def test_limit_is_a_seeded_sample(v7w_root):
    a = load_v7w(v7w_root, "train", limit=5, seed=3)
    assert [x.qid for x in a] == [x.qid for x in load_v7w(v7w_root, "train", limit=5, seed=3)]
    assert len(a) == 5
