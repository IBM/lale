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
import lale.pretty_print
import re

def _json_op_kind(jsn):
    if 'steps' in jsn and 'edges' in jsn:
        return 'Pipeline'
    elif 'steps' in jsn:
        return 'OperatorChoice'
    return 'IndividualOp'

def _get_cluster2rep(jsn):
    cluster2rep = {}
    def populate(clusters, jsn):
        kind = _json_op_kind(jsn)
        if kind in ['Pipeline', 'OperatorChoice']:
            more_clusters = [jsn['id'], *clusters]
            for step in jsn['steps']:
                populate(more_clusters, step)
        else:
            assert kind == 'IndividualOp'
            rep = jsn['id']
            for cluster in clusters:
                if cluster not in cluster2rep:
                    cluster2rep[cluster] = rep
    populate([], jsn)
    return cluster2rep

_STATE2COLOR = {
    'trained': 'white',
    'trainable': 'lightskyblue1',
    'planned': 'skyblue2'}

def _json_to_graphviz_rec(jsn, cluster2rep, is_root, dot_graph_attr):
    kind = _json_op_kind(jsn)
    if kind in ['Pipeline', 'OperatorChoice']:
        dot = graphviz.Digraph(name=f"cluster:{jsn['id']}")
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
                 tooltip=f"{jsn['id']} = ...")
        nodes = jsn['steps']
        edges = jsn['edges']
    elif kind == 'OperatorChoice':
        if is_root:
            nodes = [jsn]
        else:
            rhs = ' | '.join(s['id'] for s in jsn['steps'])
            dot.attr('graph', label='Choice', style='filled',
                     fillcolor='skyblue2', tooltip=f"{jsn['id']} = {rhs}")
            nodes = jsn['steps']
        edges = []
    else:
        assert is_root and kind == 'IndividualOp'
        nodes = [jsn]
        edges = []
    for node in nodes:
        node_kind = _json_op_kind(node)
        if node_kind in ['Pipeline', 'OperatorChoice']:
            sub_dot = _json_to_graphviz_rec(node, cluster2rep, False, {})
            dot.subgraph(sub_dot)
        else:
            assert node_kind == 'IndividualOp'
            tooltip = f"{node['id']} = {node['label']}"
            if 'hyperparams' in node:
                hps = node['hyperparams']
                if hps is not None:
                    hpss = lale.pretty_print.hyperparams_to_string(hps)
                    if len(hpss) > 255: #too long for graphviz
                        hpss = hpss[:252] + '...'
                    tooltip = f'{tooltip}({hpss})'
            attrs = {
                'style' :'filled',
                'fillcolor': _STATE2COLOR[node['state']],
                'tooltip': tooltip}
            if 'documentation_url' in node:
                attrs['URL'] = node['documentation_url']
            label1 = re.sub('(.)([A-Z][a-z]+)', r'\1-\n\2', node['label'])
            label2 = re.sub('([a-z0-9])([A-Z])', r'\1-\n\2', label1)
            label3 = re.sub(r'([^_\n-]_)([^_\n-])', r'\1-\n\2', label2)
            dot.node(node['id'], label3, **attrs)
    for edge in edges:
        src, dst = nodes[edge[0]], nodes[edge[1]]
        src_kind, dst_kind = _json_op_kind(src), _json_op_kind(dst)
        if src_kind == 'IndividualOp':
            if dst_kind == 'IndividualOp':
                dot.edge(src['id'], dst['id'])
            else:
                dot.edge(src['id'], cluster2rep[dst['id']],
                         lhead=f"cluster:{dst['id']}")
        else:
            if dst_kind == 'IndividualOp':
                dot.edge(cluster2rep[src['id']], dst['id'],
                         ltail=f"cluster:{src['id']}")
            else:
                dot.edge(cluster2rep[src['id']], cluster2rep[dst['id']],
                         ltail=f"cluster:{src['id']}",
                         lhead=f"cluster:{dst['id']}")
    return dot

def json_to_graphviz(jsn, dot_graph_attr):
    cluster2rep = _get_cluster2rep(jsn)
    dot = _json_to_graphviz_rec(jsn, cluster2rep, True, dot_graph_attr)
    return dot
