from __future__ import annotations
from typing import Dict, Set, List, Iterable, Union, Tuple

import json

Node = Dict[str, object]  # expects {"name": str, optional "children": list[Node]}


def build_abs_ancestors_maps(
    tree: Union[Node, Iterable[Node]]
) -> Tuple[Dict[str, Dict[int, Set[str]]], Dict[str, Set[str]], Dict[str, Set[str]], List[str]]:
    """
    Build absolute-level ancestor maps from a tree (or forest).

    Input:
        tree:
            - Either a single root node: {"name": ..., "children": [...]}
            - Or an iterable of such nodes (multiple roots allowed).

    Output:
        abs_ancestors:
            label -> { depth (1 = root level) -> {ancestor_name, ...} }
            Depth is "absolute" from a root: root is at depth 1, its child at depth 2, etc.
        all_ancestors:
            label -> { all ancestor names across all paths }
        parent2children:
            parent_name -> { direct child names }
        root_names:
            list of root node names (top-level category names)

    Notes:
        - Supports multiple roots.
        - Supports multiple parents in terms of *names*:
          if the same label appears in multiple places, its ancestor sets are merged.
        - Roots themselves do not appear as keys in abs_ancestors/all_ancestors,
          because they have no ancestors by definition.
    """
    roots = [tree] if isinstance(tree, dict) else list(tree)

    abs_ancestors: Dict[str, Dict[int, Set[str]]] = {}
    all_ancestors: Dict[str, Set[str]] = {}
    parent2children: Dict[str, Set[str]] = {}

    # Collect root names
    root_names: List[str] = []
    for r in roots:
        if isinstance(r, dict) and isinstance(r.get("name"), str):
            root_names.append(r["name"])

    def push_for_child(child: str, ancestor_path: List[Tuple[str, int]]) -> None:
        """
        Register ancestors for a child.

        ancestor_path: list of (ancestor_name, absolute_depth_from_root)
        """
        if child not in abs_ancestors:
            abs_ancestors[child] = {}
        if child not in all_ancestors:
            all_ancestors[child] = set()

        for anc_name, depth in ancestor_path:
            abs_ancestors[child].setdefault(depth, set()).add(anc_name)
            all_ancestors[child].add(anc_name)

    def dfs(node: Node, path: List[Tuple[str, int]]) -> None:
        """
        DFS with 'path' = list of (ancestor_name, depth).
        The current node is not included in 'path' yet; it becomes an ancestor when
        we recurse into its children.
        """
        name = node.get("name")
        if not isinstance(name, str):
            return

        children = node.get("children") or []
        if isinstance(children, list):
            for ch in children:
                if isinstance(ch, dict) and isinstance(ch.get("name"), str):
                    child_name = ch["name"]
                    parent2children.setdefault(name, set()).add(child_name)

                    # The current node 'name' becomes an ancestor at depth = len(path) + 1
                    next_path = path + [(name, len(path) + 1)]
                    push_for_child(child_name, next_path)
                    dfs(ch, next_path)

    # Start DFS from each root
    for r in roots:
        if isinstance(r, dict) and isinstance(r.get("name"), str):
            dfs(r, [])  # empty path; the root will appear as depth=1 for its children

    return abs_ancestors, all_ancestors, parent2children, root_names


# Example CLI usage
if __name__ == "__main__":
    import os
    import pickle
    from pathlib import Path

    # Assume current working directory is data_construction/
    assets_dir = Path("assets/open_images")
    os.makedirs(assets_dir, exist_ok=True)

    data_path = assets_dir / "bbox_hierarchy.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    abs_ancestors, all_ancestors, parent2children, roots = build_abs_ancestors_maps(data)
    print("Built abs_ancestors map for", len(abs_ancestors), "labels.")

    # Save as pickle into assets/
    with open(assets_dir / "abs_ancestors.pkl", "wb") as f:
        pickle.dump(abs_ancestors, f)
    print("Saved assets/abs_ancestors.pkl")

    with open(assets_dir / "all_ancestors.pkl", "wb") as f:
        pickle.dump(all_ancestors, f)
    print("Saved assets/all_ancestors.pkl")

    with open(assets_dir / "parent2children.pkl", "wb") as f:
        pickle.dump(parent2children, f)
    print("Saved assets/parent2children.pkl")

    # Simple sanity check
    sample_labels = ["Doll", "Balloon", "Coin", "Flag", "Light bulb"]
    for lbl in sample_labels:
        print(f"--- {lbl} ---")
        print("  abs_ancestors:", abs_ancestors.get(lbl))
        print("  all_ancestors:", all_ancestors.get(lbl))
        parents = [p for p, children in parent2children.items() if lbl in children]
        print("  parents:", parents)