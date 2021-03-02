# Copyright 2020 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List, Optional, Tuple

import graphviz

import lale.json_operator
import lale.pretty_print

_LALE_SKL_PIPELINE = "lale.lib.sklearn.pipeline._PipelineImpl"


def _get_cluster2reps(jsn) -> Tuple[Dict[str, str], Dict[str, str]]:
    """For each cluster (Pipeline, OperatorChoice, or higher-order IndividualOp), get two representatives (first-order IndividualOps).

    Lale visualizes composite operators using graphviz clusters. To
    visualize an edge to (from) a cluster, we need to tell graphviz a
    representative in that cluster for the edge to connect to (from).

    Parameters
    ----------
    jsn:
        JSON representation of a Lale pipeline.
        The steps of all sub-pipelines must be topologically ordered.

    Returns
    -------
    reps:
        Two dictionaries from cluster id fields to node id fields.
        Nodes in the first dictionary are roots that are furthest left
        in the visualization, suitable for incoming edges.
        Nodes in the second dictionary are roots that are furthest
        right in the visualization, suitable for outgoing edges.
    """
    cluster2root: Dict[str, str] = {}
    cluster2leaf: Dict[str, str] = {}
    node2depth: Dict[str, int] = {}

    def populate(uid: str, jsn, depth: int, clusters: List[str]) -> int:
        kind: str = lale.json_operator.json_op_kind(jsn)
        if kind == "Pipeline" or jsn["class"] == _LALE_SKL_PIPELINE:
            step2idx: Dict[str, int] = {}
            for step_idx, step_uid in enumerate(jsn["steps"].keys()):
                step2idx[step_uid] = step_idx
            if kind == "Pipeline":
                edges = jsn["edges"]
            else:
                names = list(jsn["steps"].keys())
                edges = [[names[i], names[i + 1]] for i in range(len(names) - 1)]
            for tail, head in edges:
                assert (
                    step2idx[tail] < step2idx[head]
                ), f"steps {tail} and {head} are not in topological order"
            node2preds: Dict[str, List[str]] = {
                step: [] for step in jsn["steps"].keys()
            }
            for tail, head in edges:
                node2preds[head].append(tail)
            more_clusters = [uid, *clusters]
            d_max = depth
            for step_uid, step_jsn in jsn["steps"].items():
                d_root = max(
                    [node2depth[p] for p in node2preds[step_uid]], default=depth
                )
                d_leaf = populate(step_uid, step_jsn, d_root, more_clusters)
                d_max = max(d_max, d_leaf)
        elif kind == "OperatorChoice" or "steps" in jsn:
            more_clusters = [uid, *clusters]
            d_max = depth
            for step_uid, step_jsn in jsn["steps"].items():
                d_leaf = populate(step_uid, step_jsn, depth, more_clusters)
                d_max = max(d_max, d_leaf)
        else:
            assert kind == "IndividualOp"
            d_max = depth + 1
            for cluster in clusters:
                if (
                    cluster not in cluster2root
                    or node2depth[cluster2root[cluster]] > d_max
                ):
                    cluster2root[cluster] = uid
                if (
                    cluster not in cluster2leaf
                    or node2depth[cluster2leaf[cluster]] < d_max
                ):
                    cluster2leaf[cluster] = uid
        node2depth[uid] = d_max
        return d_max

    populate("(root)", jsn, 0, [])
    return cluster2root, cluster2leaf


_STATE2COLOR = {"trained": "white", "trainable": "lightskyblue1", "planned": "skyblue2"}


def _indiv_op_tooltip(uid, jsn) -> str:
    assert lale.json_operator.json_op_kind(jsn) == "IndividualOp"
    tooltip = f"{uid} = {jsn['label']}"
    if "hyperparams" in jsn:
        hps = jsn["hyperparams"]
        if hps is not None:
            steps: Optional[Dict[str, Any]]
            if "steps" in jsn:
                steps = {step_uid: step_uid for step_uid in jsn["steps"]}
            else:
                steps = None
            hp_string = lale.pretty_print.hyperparams_to_string(hps, steps)
            if len(hp_string) > 255:  # too long for graphviz
                hp_string = hp_string[:252] + "..."
            tooltip = f"{tooltip}({hp_string})"
    return tooltip


def _json_to_graphviz_rec(uid, jsn, cluster2reps, is_root, dot_graph_attr):
    kind = lale.json_operator.json_op_kind(jsn)
    if kind in ["Pipeline", "OperatorChoice"] or "steps" in jsn:
        dot = graphviz.Digraph(name=f"cluster:{uid}")
    else:
        dot = graphviz.Digraph()
    if is_root:
        dot.attr(
            "graph",
            {**dot_graph_attr, "rankdir": "LR", "compound": "true", "nodesep": "0.1"},
        )
        dot.attr("node", fontsize="11", margin="0.06,0.03")
    if kind == "Pipeline":
        dot.attr(
            "graph",
            label="",
            style="rounded,filled",
            fillcolor=_STATE2COLOR[jsn["state"]],
            tooltip=f"{uid} = ...",
        )
        nodes = jsn["steps"]
        edges = jsn["edges"]
    else:
        if is_root:
            nodes = {"(root)": jsn}
            edges = []
        elif kind == "OperatorChoice":
            rhs = " | ".join(jsn["steps"].keys())
            dot.attr(
                "graph",
                label="Choice",
                style="filled",
                fillcolor=_STATE2COLOR[jsn["state"]],
                tooltip=f"{uid} = {rhs}",
            )
            nodes = jsn["steps"]
            edges = []
        else:
            assert kind == "IndividualOp" and "steps" in jsn
            dot.attr(
                "graph",
                label=jsn.get("viz_label", jsn["label"]),
                style="filled",
                fillcolor=_STATE2COLOR[jsn["state"]],
                tooltip=_indiv_op_tooltip(uid, jsn),
            )
            if "documentation_url" in jsn:
                dot.attr("graph", URL=jsn["documentation_url"])
            nodes = jsn["steps"]
            if jsn["class"] == _LALE_SKL_PIPELINE:
                names = list(nodes.keys())
                edges = [[names[i], names[i + 1]] for i in range(len(names) - 1)]
            else:
                edges = []
    for step_uid, step_jsn in nodes.items():
        node_kind = lale.json_operator.json_op_kind(step_jsn)
        if node_kind in ["Pipeline", "OperatorChoice"] or "steps" in step_jsn:
            sub_dot = _json_to_graphviz_rec(step_uid, step_jsn, cluster2reps, False, {})
            dot.subgraph(sub_dot)
        else:
            assert node_kind == "IndividualOp"
            tooltip = _indiv_op_tooltip(step_uid, step_jsn)
            attrs = {
                "style": "filled",
                "fillcolor": _STATE2COLOR[step_jsn["state"]],
                "tooltip": tooltip,
            }
            if "documentation_url" in step_jsn:
                attrs["URL"] = step_jsn["documentation_url"]
            label0 = step_jsn.get("viz_label", step_jsn["label"])
            if "\n" in label0:
                label3 = label0
            else:
                label1 = re.sub("(.)([A-Z][a-z]+)", r"\1-\n\2", label0)
                label2 = re.sub("([a-z0-9])([A-Z])", r"\1-\n\2", label1)
                label3 = re.sub(r"([^_\n-]_)([^_\n-])", r"\1-\n\2", label2)
            dot.node(step_uid, label3, **attrs)
    cluster2root, cluster2leaf = cluster2reps
    for tail, head in edges:
        tail_is_cluster = "steps" in nodes[tail]
        head_is_cluster = "steps" in nodes[head]
        if tail_is_cluster:
            if head_is_cluster:
                dot.edge(
                    cluster2leaf[tail],
                    cluster2root[head],
                    ltail=f"cluster:{tail}",
                    lhead=f"cluster:{head}",
                )
            else:
                dot.edge(cluster2leaf[tail], head, ltail=f"cluster:{tail}")
        else:
            if head_is_cluster:
                dot.edge(tail, cluster2root[head], lhead=f"cluster:{head}")
            else:
                dot.edge(tail, head)
    return dot


def json_to_graphviz(jsn, ipython_display, dot_graph_attr):
    cluster2reps = _get_cluster2reps(jsn)
    dot = _json_to_graphviz_rec("(root)", jsn, cluster2reps, True, dot_graph_attr)
    if ipython_display:
        import IPython.display

        IPython.display.display(dot)
        return None
    return dot
