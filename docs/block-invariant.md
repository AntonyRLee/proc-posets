# A computable block invariant

**Provenance.** This was Appendix B ("A computable block invariant") of *On the Geometry of Process
Mining* (Lee, Tiňo, Styles), moved here on 2026-08-19. It is implementation documentation, not part
of the paper's argument: the paper's canonical-tiling proposition identifies blocks *up to
isomorphism*, which is all the injectivity of the tiling map needs. An implementation additionally
needs a concrete representative, and that is what this document specifies. Nothing in the paper
depends on it.

## The problem

The tiling map sends a variant to a word over an alphabet of blocks. Blocks are identified up to
isomorphism, so an implementation must be able to decide isomorphism, and — to use blocks as
dictionary keys — must produce a canonical string `c(β)` with

    c(β) = c(β′)  if and only if  β and β′ are isomorphic.

That is, a *complete invariant*. The construction below builds one by structural induction over the
modular decomposition tree `T(D)`.

## The construction

Encode each node `t` of `T(D)` by a string `c(t)`, by induction on height.

| node type | code |
|---|---|
| leaf | its activity label `ℓ(t)` |
| series, children `t₁ … t_r` in their order | the ordered tuple `⟨c(t₁), …, c(t_r)⟩` |
| parallel, children `t₁ … t_r` | the multiset `{{c(t₁), …, c(t_r)}}`, sorted under a fixed total order on strings |
| prime, with quotient `Q` | see below |

For a **prime** node with quotient `Q` on `q` vertices, take the lexicographic minimum, over all
numberings `σ` of `Q`'s vertices that linearly extend its order, of the combined string

    ( cmp(Q, σ) ; c(t_{σ⁻¹(1)}), …, c(t_{σ⁻¹(q)}) )

where `cmp(Q, σ)` is the comparability matrix of `Q` written out in the order `σ`. The minimising `σ`
fixes the vertex order `v₁ … v_q`.

### Why the minimisation is joint, and not over the quotient order alone

This is the subtle point and the one an implementation is most likely to get wrong. The minimisation
must range over the full combined string — the comparability matrix *together with* the children's
codes — rather than over the quotient order by itself.

The reason is that a prime quotient may have a large automorphism group. The **crown on six points**
has automorphism group `S₃`. An order-only choice of numbering is therefore ambiguous: several
numberings extend the order and are indistinguishable by the order alone, and picking among them
arbitrarily makes the code depend on that arbitrary choice.

Minimising jointly removes the ambiguity, because any residual tie relates positions that are both
order-automorphic *and* carry equal children codes — so the recorded tuple is invariant either way.

## Completeness

**Claim.** `c(t) = c(t′)` if and only if the subtrees rooted at `t` and `t′` are isomorphic.

*Necessity.* A label-preserving isomorphism maps numberings of `Q` bijectively to numberings of `Q′`
and preserves the key, so the two minima agree; equal children follow by the induction hypothesis.

*Sufficiency.* Equal strings share an outermost constructor, hence a node type.

- **Series**: agrees componentwise.
- **Parallel**: equal sorted multisets, so a bijection matches children with equal codes.
- **Prime**: equal `cmp` under the minimising numberings, which reconstruct isomorphic labelled
  quotients aligning the vertex indices.

In each case the induction hypothesis upgrades equal children codes to child isomorphisms, which
assemble into a subtree isomorphism.

Labels enter only at leaves and are compared verbatim, so **repeated activity labels are no
exception** — the invariant handles them without special-casing.

Hence `c` is a complete invariant, and each maximal non-series node yields the block symbol `c(β)`
that the canonical-tiling proposition requires.

## Implementation notes

- The prime case is the only one whose cost is not linear in the subtree: it minimises over
  numberings extending the quotient order. For prime quotients of realistic size this is
  acceptable, but it is the place to look first if block encoding becomes a bottleneck.
- The fixed total order on strings used by the parallel case must be stable across a whole
  comparison study, since block symbols are dictionary keys shared between models.
- The crown case above is worth keeping as a regression test: an order-only implementation passes
  every series/parallel test and fails only on a prime quotient with a non-trivial automorphism
  group.
