"""Sidebar tree grouping models by profile."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rich.text import Text
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from vllmctl.service import CatalogEntry, ProfileView


@dataclass(frozen=True)
class ProfileNodeData:
    """Marker payload for a profile group node."""

    name: str


NodeData = CatalogEntry | ProfileNodeData
Signature = tuple[tuple[str, tuple[str, ...]], ...]


class ModelsTree(Tree[NodeData]):
    """Sidebar tree: profile group headers with model leaves underneath.

    Steady-state refreshes only update label text on existing nodes so the
    cursor never resets. A full rebuild only happens when the catalog
    structure changes.
    """

    ICON_NODE = "▸ "
    ICON_NODE_EXPANDED = "▾ "

    # The sidebar is fixed at 42 cols in app.py. Subtract border (2) and
    # tree padding (2). Leaves 38 cols of content width.
    CONTENT_WIDTH = 38
    PROFILE_ICON_WIDTH = 2  # "▾ "
    LEAF_PREFIX_WIDTH = 5  # indent (guide_depth=3) + badge "● "
    PORT_RIGHT_MARGIN = 5  # ports stop this many cols before the right edge

    DEFAULT_CSS = """
    ModelsTree {
        height: 1fr;
        border: round $accent;
        padding: 0 1;
    }
    ModelsTree:focus {
        border: round $primary;
    }
    ModelsTree > .tree--cursor {
        background: $accent;
        color: black;
        text-style: bold;
    }
    ModelsTree:focus > .tree--cursor {
        background: $primary;
        color: black;
        text-style: bold;
    }
    """

    def __init__(self) -> None:
        super().__init__("Models")
        self.show_root = False
        self.show_guides = False
        self.guide_depth = 3
        self._signature: Signature = ()
        self._profile_nodes: dict[str, TreeNode[NodeData]] = {}
        self._model_nodes: dict[str, TreeNode[NodeData]] = {}

    def on_mount(self) -> None:
        self.border_title = "Models"

    def render_profiles(self, views: Iterable[ProfileView]) -> None:
        views = [v for v in views if v.entries]
        signature = _signature_of(views)
        if signature == self._signature and self._model_nodes:
            self._update_labels(views)
            return
        self._signature = signature
        self._rebuild(views)

    def _update_labels(self, views: list[ProfileView]) -> None:
        for view in views:
            profile_node = self._profile_nodes.get(view.name)
            if profile_node is not None:
                profile_node.set_label(_profile_label(view))
            for entry in view.entries:
                node = self._model_nodes.get(entry.name)
                if node is not None:
                    node.set_label(_model_label(entry))
                    node.data = entry

    def _rebuild(self, views: list[ProfileView]) -> None:
        previous_model = self.selected_model_name
        previous_profile = self.selected_profile_name

        self.root.remove_children()
        self._profile_nodes.clear()
        self._model_nodes.clear()

        for view in views:
            profile_node = self.root.add(
                _profile_label(view),
                data=ProfileNodeData(name=view.name),
                expand=True,
            )
            self._profile_nodes[view.name] = profile_node
            for entry in view.entries:
                leaf = profile_node.add_leaf(_model_label(entry), data=entry)
                self._model_nodes[entry.name] = leaf

        if previous_model is not None and previous_model in self._model_nodes:
            self.select_node(self._model_nodes[previous_model])
        elif previous_profile is not None and previous_profile in self._profile_nodes:
            self.select_node(self._profile_nodes[previous_profile])

    @property
    def selected_model_name(self) -> str | None:
        node = self.cursor_node
        if node is None or not isinstance(node.data, CatalogEntry):
            return None
        return node.data.name

    @property
    def selected_profile_name(self) -> str | None:
        node = self.cursor_node
        if node is None or not isinstance(node.data, ProfileNodeData):
            return None
        return node.data.name


def _signature_of(views: list[ProfileView]) -> Signature:
    return tuple((v.name, tuple(e.name for e in v.entries)) for v in views)


def _profile_label(view: ProfileView) -> Text:
    label = Text()
    style = "dim bold" if view.is_general else "bold"
    label.append(view.name, style=style)
    counter = f"{view.running_count}/{view.total_count}"
    available = ModelsTree.CONTENT_WIDTH - ModelsTree.PROFILE_ICON_WIDTH
    padding = max(1, available - len(view.name) - len(counter))
    label.append(" " * padding)
    label.append(counter, style="dim")
    return label


def _model_label(entry: CatalogEntry) -> Text:
    label = Text()
    if entry.is_broken:
        label.append("! ", style="bold red")
        label.append(entry.name)
        return label
    status = entry.status
    assert status is not None
    if status.running:
        label.append("● ", style="bold green")
    elif status.stale_pid_file:
        label.append("▴ ", style="bold yellow")
    else:
        label.append("◌ ", style="bright_black")
    label.append(entry.name)
    if status.metrics_port:
        port = f":{status.metrics_port}"
        available = ModelsTree.CONTENT_WIDTH - ModelsTree.LEAF_PREFIX_WIDTH - ModelsTree.PORT_RIGHT_MARGIN
        padding = max(1, available - len(entry.name) - len(port))
        label.append(" " * padding)
        label.append(port, style="dim")
    return label
