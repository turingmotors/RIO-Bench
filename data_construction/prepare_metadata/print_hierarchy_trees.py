from typing import Dict, List, Iterable, Union, Optional, Set
from pathlib import Path

Node = Dict[str, object]  # expects {"name": str, "children": List[Node]}

# -----------------------------
# Pretty tree printing (box-drawing)
# -----------------------------
def tree_to_lines(
    tree: Union[Node, Iterable[Node]],
    sort_children: bool = True,
    max_depth: Optional[int] = None,
    show_level: bool = False,
    show_root_level: int = 1,
) -> List[str]:
    """
    Convert a hierarchy (single dict or list of dicts) into pretty text lines using box-drawing chars.
    - sort_children: sort siblings by name for stable output
    - max_depth: cut off deeper levels (None = unlimited)
    - show_level: prefix each node with [L=depth] (root depth = show_root_level)
    """
    roots = [tree] if isinstance(tree, dict) else list(tree)

    def get_name(n) -> str:
        return str(n.get("name", ""))

    def get_children(n) -> List[Node]:
        ch = n.get("children") or []
        ch = [c for c in ch if isinstance(c, dict)]
        if sort_children:
            ch = sorted(ch, key=lambda x: get_name(x))
        return ch

    lines: List[str] = []

    def fmt_prefix(prefix_stack: List[bool]) -> str:
        # prefix_stack: for each ancestor level, True if there are more siblings afterwards
        parts = []
        for has_more in prefix_stack[:-1]:
            parts.append("│   " if has_more else "    ")
        if not prefix_stack:
            return ""
        return "".join(parts) + ("├── " if prefix_stack[-1] else "└── ")

    def dfs(node: Node, prefix_stack: List[bool], depth: int):
        if max_depth is not None and depth > max_depth:
            return
        name = get_name(node)
        prefix = fmt_prefix(prefix_stack)
        label = f"[L={depth}] {name}" if show_level else name
        if prefix_stack:
            lines.append(prefix + label)
        else:
            # root line (no prefix)
            lines.append(label)
        children = get_children(node)
        if max_depth is not None and depth >= max_depth:
            return
        for i, ch in enumerate(children):
            has_more = (i < len(children) - 1)
            dfs(ch, prefix_stack + [has_more], depth + 1)

    for r in roots:
        dfs(r, [], show_root_level)

    return lines


def write_tree_txt(
    tree: Union[Node, Iterable[Node]],
    out_path: str,
    **kwargs
) -> None:
    """Write pretty tree text into a file."""
    lines = tree_to_lines(tree, **kwargs)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# -----------------------------
# From parent->children adjacency
# -----------------------------
def adjacency_to_forest(parent2children: Dict[str, Set[str]]) -> List[Node]:
    """
    Build a forest (list of root nodes) from a parent->children adjacency.
    - Detects roots as nodes that never appear as a child.
    - Produces a minimal Node structure: {"name": str, "children": [...]}
    """
    parents = set(parent2children.keys())
    children = set().union(*parent2children.values()) if parent2children else set()
    all_nodes = parents | children
    roots = sorted(all_nodes - children)

    # cache nodes
    name2node: Dict[str, Node] = {n: {"name": n, "children": []} for n in all_nodes}
    for p, chs in parent2children.items():
        pnode = name2node.setdefault(p, {"name": p, "children": []})
        for c in sorted(chs):
            cnode = name2node.setdefault(c, {"name": c, "children": []})
            pnode["children"].append(cnode)

    return [name2node[r] for r in roots]


def write_adjacency_txt(
    parent2children: Dict[str, Set[str]],
    out_path: str,
    **kwargs
) -> None:
    """Build a forest from adjacency and save as pretty tree text."""
    forest = adjacency_to_forest(parent2children)
    write_tree_txt(forest, out_path, **kwargs)


if __name__ == "__main__":
    import pickle

    # Assume current working directory is data_construction/
    assets_dir = Path("assets/open_images")
    parent2children_path = assets_dir / "parent2children.pkl"

    with parent2children_path.open("rb") as f:
        parent2children = pickle.load(f)

    # Only this one output, saved into assets/
    out_path = assets_dir / "hierarchy_from_adjacency.txt"
    write_adjacency_txt(
        parent2children,
        out_path,
        sort_children=True,
        max_depth=None,     # or e.g., 4
        show_level=True,    # print [L=depth]
        show_root_level=1   # root has L=1
    )
    print(f"Saved {out_path}")