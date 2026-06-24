"""
net3.edit — interactive graph editor.

Backend (GUI-free, fully testable):
    from net3.edit import GraphEditor
    ed = GraphEditor.from_gpickle("graph.gpickle", mask_path="mask.tif")
    ed.toggle_selection(node_id)
    ed.delete_selected()
    ed.undo()
    ed.save("graph_edited.gpickle")

Frontend (requires the [gui] extra: napari + qtpy):
    from net3.edit.app import run_editor
    run_editor("graph.gpickle", mask_path="mask.tif")

CLI:
    net3 edit graph.gpickle --mask mask.tif
"""

from .core import GraphEditor

__all__ = ["GraphEditor"]
