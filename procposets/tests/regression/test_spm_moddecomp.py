from procposets.moddecomp import Leaf, Parallel, Prime, Series, decompose, tiling
from procposets.poset import from_dag, leaf, n_poset, par, then


def test_leaf():
    t = decompose(leaf("a"))
    assert isinstance(t, Leaf)
    assert t.canonical() == "a"


def test_parallel_is_atomic_block():
    t = decompose(par(leaf("a"), leaf("b")))
    assert isinstance(t, Parallel)
    assert t.canonical() == "(a * b)"  # commutative -> sorted


def test_series_is_a_chain():
    t = decompose(then(leaf("a"), leaf("b")))
    assert isinstance(t, Series)
    assert t.canonical() == "(a ; b)"


def test_series_of_parallel():
    t = decompose(then(leaf("a"), par(leaf("b"), leaf("c")), leaf("d")))
    assert isinstance(t, Series)
    assert t.canonical() == "(a ; (b * c) ; d)"


def test_parallel_of_series_stays_atomic():
    # concurrency wraps a sequence: the whole thing is one atomic (parallel) block
    t = decompose(par(then(leaf("a"), leaf("b")), leaf("c")))
    assert isinstance(t, Parallel)
    assert t.canonical() == "((a ; b) * c)"


def test_n_poset_is_prime():
    t = decompose(n_poset())
    assert isinstance(t, Prime)
    assert t.canonical() == "N{a<c, b<c, b<d}"


def test_concurrency_differs_from_sequence():
    assert tiling(par(leaf("a"), leaf("b"))) != tiling(then(leaf("a"), leaf("b")))


def test_from_dag_nshape_is_prime():
    assert isinstance(decompose(from_dag([("a", "c"), ("b", "c"), ("b", "d")])), Prime)


def test_from_dag_nshape_then_f_is_series_over_prime():
    t = decompose(from_dag([("a", "c"), ("b", "c"), ("b", "d"), ("c", "f"), ("d", "f")]))
    assert isinstance(t, Series)
    assert any(isinstance(c, Prime) for c in t.children)
