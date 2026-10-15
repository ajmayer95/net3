"""
net3.edit — interactive graph and mask editors.

Backends (GUI-free, fully testable):
    from net3.edit import GraphEditor
    ed = GraphEditor.from_gpickle("graph.gpickle", mask_path="mask.tif")
    ed.toggle_selection(node_id)
    ed.delete_selected()
    ed.undo()
    ed.save("graph_edited.gpickle")

    from net3.edit import MaskEditor
    me = MaskEditor.from_paths("mask.tif", image_path="raw.tif")
    me.threshold(0.5, remove_small=500, fill_holes=200)
    me.apply_median_filter(radius=3)
    me.undo()
    me.save_mask("mask_edited.tif")

Frontends (require the [gui] extra: napari + qtpy):
    from net3.edit.app import run_editor
    run_editor("graph.gpickle", mask_path="mask.tif")

    from net3.edit.mask_gui import open_mask_editor
    open_mask_editor(mask="mask.tif", image="raw.tif")

CLI:
    net3 edit graph.gpickle --mask mask.tif
    net3 edit-mask mask.tif [--image raw.tif] [--probs probs.tif]
"""

from .core import GraphEditor
from .mask import MaskEditor

__all__ = ["GraphEditor", "MaskEditor"]
