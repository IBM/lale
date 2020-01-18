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

import graphviz
import lale.json_operator
import lale.pretty_print
import re
from typing import Any, Dict, List, Tuple

def _get_cluster2reps(jsn) -> Tuple[Dict[str, str], Dict[str, str]]:
    """For each cluster (Pipeline or OperatorChoice), get two representatives (IndividualOps).

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
        if kind == 'Pipeline':
            step2idx: Dict[str, int] = {}
            for step_idx, step_uid in enumerate(jsn['steps'].keys()):
                step2idx[step_uid] = step_idx
            for tail, head in jsn['edges']:
                assert step2idx[tail] < step2idx[head], \
                    f'steps {tail} and {head} are not in topological order'
            node2preds: Dict[str, List[str]] = {
                step: [] for step in jsn['steps'].keys()}
            for tail, head in jsn['edges']:
                node2preds[head].append(tail)
            more_clusters = [uid, *clusters]
            d_max = depth
            for step_uid, step_jsn in jsn['steps'].items():
                d_root = max([node2depth[p] for p in node2preds[step_uid]],
                             default=depth)
                d_leaf = populate(step_uid, step_jsn, d_root, more_clusters)
                d_max = max(d_max, d_leaf)
        elif kind == 'OperatorChoice':
            more_clusters = [uid, *clusters]
            d_max = depth
            for step_uid, step_jsn in jsn['steps'].items():
                d_leaf = populate(step_uid, step_jsn, depth, more_clusters)
                d_max = max(d_max, d_leaf)
        else:
            assert kind == 'IndividualOp'
            d_max = depth + 1
            for cluster in clusters:
                if (cluster not in cluster2root or
                    node2depth[cluster2root[cluster]] > d_max):
                    cluster2root[cluster] = uid
                if (cluster not in cluster2leaf or
                    node2depth[cluster2leaf[cluster]] < d_max):
                    cluster2leaf[cluster] = uid
        node2depth[uid] = d_max
        return d_max
    populate('(root)', jsn, 0, [])
    return cluster2root, cluster2leaf

_STATE2COLOR = {
    'trained': 'white',
    'trainable': 'lightskyblue1',
    'planned': 'skyblue2'}

def _json_to_graphviz_rec(uid, jsn, cluster2reps, is_root, dot_graph_attr):
    kind = lale.json_operator.json_op_kind(jsn)
    if kind in ['Pipeline', 'OperatorChoice']:
        dot = graphviz.Digraph(name=f"cluster:{uid}")
    else:
        dot = graphviz.Digraph()
    if is_root:
        dot.attr('graph', {**dot_graph_attr,
                           'rankdir': 'LR', 'compound': 'true',
                           'nodesep': '0.1'})
        dot.attr('node', fontsize='11', margin='0.06,0.03')
    if kind == 'Pipeline':
        dot.attr('graph', label='', style='rounded,filled',
                 fillcolor=_STATE2COLOR[jsn['state']],
                 tooltip=f"{uid} = ...")
        nodes = jsn['steps']
        edges = jsn['edges']
    elif kind == 'OperatorChoice':
        if is_root:
            nodes = {'(root)': jsn}
        else:
            rhs = ' | '.join(jsn['steps'].keys())
            dot.attr('graph', label='Choice', style='filled',
                     fillcolor='skyblue2', tooltip=f"{uid} = {rhs}")
            nodes = jsn['steps']
        edges = []
    else:
        assert is_root and kind == 'IndividualOp'
        nodes = {'(root)': jsn}
        edges = []
    for step_uid, step_jsn in nodes.items():
        node_kind = lale.json_operator.json_op_kind(step_jsn)
        if node_kind in ['Pipeline', 'OperatorChoice']:
            sub_dot = _json_to_graphviz_rec(
                step_uid, step_jsn, cluster2reps, False, {})
            dot.subgraph(sub_dot)
        else:
            assert node_kind == 'IndividualOp'
            tooltip = f"{step_uid} = {step_jsn['label']}"
            if 'hyperparams' in step_jsn:
                hps = step_jsn['hyperparams']
                if hps is not None:
                    hpss = lale.pretty_print.hyperparams_to_string(hps)
                    if len(hpss) > 255: #too long for graphviz
                        hpss = hpss[:252] + '...'
                    tooltip = f'{tooltip}({hpss})'
            attrs = {
                'style' :'filled',
                'fillcolor': _STATE2COLOR[step_jsn['state']],
                'tooltip': tooltip}
            if 'documentation_url' in step_jsn:
                attrs['URL'] = step_jsn['documentation_url']
            label1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\n\2', step_jsn['label'])
            label2 = re.sub('([a-z0-9])([A-Z])', r'\1-\n\2', label1)
            label3 = re.sub(r'([^_\n-]_)([^_\n-])', r'\1-\n\2', label2)
            dot.node(step_uid, label3, **attrs)
    cluster2root, cluster2leaf = cluster2reps
    for tail, head in edges:
        tail_kind = lale.json_operator.json_op_kind(nodes[tail])
        head_kind = lale.json_operator.json_op_kind(nodes[head])
        if tail_kind == 'IndividualOp':
            if head_kind == 'IndividualOp':
                dot.edge(tail, head)
            else:
                dot.edge(tail, cluster2root[head], lhead=f"cluster:{head}")
        else:
            if head_kind == 'IndividualOp':
                dot.edge(cluster2leaf[tail], head, ltail=f"cluster:{tail}")
            else:
                dot.edge(cluster2leaf[tail], cluster2root[head],
                         ltail=f"cluster:{tail}", lhead=f"cluster:{head}")
    return dot

def json_to_graphviz(jsn, dot_graph_attr):
    cluster2reps = _get_cluster2reps(jsn)
    dot = _json_to_graphviz_rec(
        '(root)', jsn, cluster2reps, True, dot_graph_attr)
    return dot
