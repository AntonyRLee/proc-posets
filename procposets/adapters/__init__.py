"""procposets.adapters -- pm4py model adapters ([pm4py] extra).

- inbound  : pm4py model -> LMGraph (then cospan.engine.extract_signature)
- outbound : procposets Model (list[(Poset, weight)]) -> pm4py ProcessTree /
             Petri net / stochastic language, plus EMD and replay fitness.
"""
from .from_bpmn import lmgraph_from_bpmn, lmgraph_from_bpmn_diagrams
from .from_process_tree import lmgraph_from_process_tree, lmgraph_from_process_trees
from .outbound import (
    alignment_fitness,
    emd,
    log_language,
    pairwise_emd,
    to_dataframe,
    to_language,
    to_petri_net,
    to_process_tree,
    token_fitness,
)

__all__ = [
    "lmgraph_from_bpmn", "lmgraph_from_bpmn_diagrams",
    "lmgraph_from_process_tree", "lmgraph_from_process_trees",
    "to_process_tree", "to_petri_net", "to_language", "log_language",
    "emd", "pairwise_emd", "to_dataframe", "alignment_fitness", "token_fitness",
]
