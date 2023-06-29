import json
from param_bench.train.compute.python.tools.eg_replay_utils import (
    get_input_tensors,
    get_output_tensors,
    is_qualified
)
from param_bench.train.compute.python.tools.execution_graph import ExecutionGraph

skip_node_names = ["DataLoader"]

def extract_nodes(root):
    replay_nodes = []

    def dfs_traverse(node):
        nonlocal replay_nodes

        for child in node.children:
            try:
                if any(x in child.name for x in skip_node_names):
                    continue
                print('1', child.id)
                if is_qualified(child):
                    replay_nodes.append(child)
                else:
                    print('2', child.id)
                    dfs_traverse(child)
            except Exception as e:
                print(f"Graph parse error: {e}, node id: {child.id}")
                exit(1)

    dfs_traverse(root)
    replay_nodes = sorted(replay_nodes, key=lambda x: x.id)
    print("#Operators to execute:", len(replay_nodes))
    return replay_nodes

def main():
    with open('/zhang-x3/users/ml2585/eg_logs/resnet_dist_eg_0.json', "r") as f:
        eg = ExecutionGraph(json.load(f))

    nodes = eg.get_nodes(clean=True)
    root = nodes[1]  # 1-base

    replay_nodes = extract_nodes(root)

    nodes_tensors = {}
    for node in replay_nodes:
        nodes_tensors[node] = set()
        for _, t_id, _ in get_input_tensors(node):
            t_id = tuple(list(t_id)[:5])
            nodes_tensors[node].add(t_id)
        for _, t_id, _ in get_output_tensors(node):
            t_id = tuple(list(t_id)[:5])
            nodes_tensors[node].add(t_id)
    

if __name__ == "__main__":
    main()
