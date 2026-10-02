"""Reusable menu state for slash completion and command pickers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MenuItem:
    key: str
    label: str
    description: str = ""
    suffix: str = ""
    search_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChecklistResult:
    selected_key: str
    disabled_keys: frozenset[str]


@dataclass(slots=True)
class MenuState:
    kind: str
    title: str
    items: list[MenuItem] = field(default_factory=list)
    index: int = 0
    hint: str = "Up/Down select · Enter confirm · Esc cancel"
    filterable: bool = False
    horizontal_keys: frozenset[str] = frozenset()
    summary_lines: list[str] = field(default_factory=list)
    detail_lines: list[str] = field(default_factory=list)
    detail_label: str = ""
    detail_scroll: int = 0
    query: str = ""
    note_allowed: bool = False
    note_active: bool = False
    note_text: str = ""
    disabled_keys: set[str] = field(default_factory=set)
    _source_items: list[MenuItem] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self._source_items:
            self._source_items = list(self.items)

    @property
    def selected(self) -> MenuItem | None:
        if not self.items:
            return None
        self.index = max(0, min(self.index, len(self.items) - 1))
        return self.items[self.index]

    def move(self, amount: int) -> None:
        if not self.items:
            self.index = 0
            return
        self.index = (self.index + amount) % len(self.items)

    @property
    def horizontal_enabled(self) -> bool:
        selected = self.selected
        return bool(selected and selected.key in self.horizontal_keys)

    def toggle_selected(self) -> None:
        selected = self.selected
        if selected is None:
            return
        if selected.key in self.disabled_keys:
            self.disabled_keys.remove(selected.key)
        else:
            self.disabled_keys.add(selected.key)

    def apply_filter(self, query: str) -> None:
        """Filter and rank source items while retaining a matching selection."""

        selected_key = self.selected.key if self.selected is not None else None
        self.query = query.strip()
        needle = self.query.casefold()
        if not self.filterable or not needle:
            ranked = list(self._source_items)
        else:
            matches: list[tuple[int, int, MenuItem]] = []
            for source_index, item in enumerate(self._source_items):
                values = (
                    item.key,
                    item.label,
                    item.description,
                    item.suffix,
                    *item.search_terms,
                )
                haystacks = tuple(value.casefold() for value in values if value)
                if any(value == needle for value in haystacks):
                    rank = 0
                elif any(value.startswith(needle) for value in haystacks):
                    rank = 1
                elif any(needle in value for value in haystacks):
                    rank = 2
                else:
                    continue
                matches.append((rank, source_index, item))
            matches.sort(key=lambda match: (match[0], match[1]))
            ranked = [item for _, _, item in matches]

        self.items = ranked
        self.index = next(
            (
                item_index
                for item_index, item in enumerate(self.items)
                if item.key == selected_key
            ),
            0,
        )

    def replace_items(self, items: list[MenuItem]) -> None:
        """Replace a live catalog without moving a still-present selection."""

        selected_key = self.selected.key if self.selected is not None else None
        self._source_items = list(items)
        self.apply_filter(self.query)
        if selected_key is not None:
            self.index = next(
                (
                    item_index
                    for item_index, item in enumerate(self.items)
                    if item.key == selected_key
                ),
                self.index,
            )

    def scroll_detail(self, amount: int, *, viewport: int) -> None:
        maximum = max(0, len(self.detail_lines) - max(1, viewport))
        self.detail_scroll = max(0, min(self.detail_scroll + amount, maximum))

    def visible_detail(self, limit: int) -> tuple[int, list[str]]:
        maximum = max(0, len(self.detail_lines) - max(1, limit))
        self.detail_scroll = max(0, min(self.detail_scroll, maximum))
        return self.detail_scroll, self.detail_lines[
            self.detail_scroll : self.detail_scroll + max(0, limit)
        ]

    def visible(self, limit: int = 8) -> tuple[int, list[MenuItem]]:
        if len(self.items) <= limit:
            return 0, self.items
        half = limit // 2
        start = max(0, min(self.index - half, len(self.items) - limit))
        return start, self.items[start : start + limit]
