# -*- coding: utf-8 -*-
"""
Copyright (c) 2020, University of Southampton
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

from pathlib import Path


class DisplayablePath(object):
    display_filename_prefix_middle = "├──"
    display_filename_prefix_last = "└──"
    display_parent_prefix_middle = "    "
    display_parent_prefix_last = "│   "

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    @classmethod
    def make_tree(
        cls,
        root,
        parent=None,
        is_last=False,
        criteria=None,
        max_depth=100,
        this_depth=0,
    ):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(
            list(path for path in root.iterdir() if criteria(path)),
            key=lambda s: str(s).lower(),
        )
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir() and this_depth < max_depth:
                if not is_last:
                    this_depth += 1
                yield from cls.make_tree(
                    path,
                    parent=displayable_root,
                    is_last=is_last,
                    criteria=criteria,
                    max_depth=max_depth,
                    this_depth=this_depth,
                )
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def show_tree(cls, root=None, max_depth=100):
        paths = []
        if root is not None:
            paths = cls.make_tree(root, max_depth=max_depth)
        else:
            paths = cls.make_tree(cls.path, max_depth=max_depth)
        for path in paths:
            if path.depth < max_depth:
                print(path.displayable())

    @classmethod
    def _default_criteria(cls, path):
        return True

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (
            self.display_filename_prefix_last
            if self.is_last
            else self.display_filename_prefix_middle
        )

        parts = ["{!s} {!s}".format(_filename_prefix, self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(
                self.display_parent_prefix_middle
                if parent.is_last
                else self.display_parent_prefix_last
            )
            parent = parent.parent

        return "".join(reversed(parts))
