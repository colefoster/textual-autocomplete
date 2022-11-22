from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Input

from textual_autocomplete._autocomplete import AutoComplete, Candidate


def get_results(value: str, cursor_position: int) -> list[Candidate]:
    candidates = [
        Candidate("f", "foo", "abc"),
        Candidate("p", "bar", "def"),
        Candidate("f", "baz", "ghi"),
    ]
    return [c for c in candidates if value in c.main]


class CompletionExample(App):
    CSS_PATH = "basic.css"

    def compose(self) -> ComposeResult:
        yield Input(id="search-box")
        yield AutoComplete(linked_input="#search-box", get_results=get_results)


app = CompletionExample()
if __name__ == '__main__':
    app.run()
