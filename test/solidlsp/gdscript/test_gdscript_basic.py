import os
import shutil

import pytest

from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_config import Language
from solidlsp.ls_utils import SymbolUtils


def _has_godot() -> bool:
    env_candidates = ["SERENA_GODOT_BIN", "GODOT_BIN", "GODOT4_BIN"]
    for env_name in env_candidates:
        value = os.environ.get(env_name)
        if not value:
            continue
        candidate = value if os.path.isabs(value) else shutil.which(value)
        if candidate and os.path.exists(candidate):
            return True
    for executable in ("godot", "godot4", "godot4-bin"):
        if shutil.which(executable):
            return True
    return False


if not _has_godot():  # pragma: no cover - depends on the developer environment
    pytest.skip("Godot binary not found; skipping GDScript language server tests", allow_module_level=True)


@pytest.mark.gdscript
class TestGDScriptLanguageServer:
    @pytest.mark.parametrize("language_server", [Language.GDSCRIPT], indirect=True)
    def test_symbol_tree_contains_core_scripts(self, language_server: SolidLanguageServer) -> None:
        symbols = language_server.request_full_symbol_tree()
        assert SymbolUtils.symbol_tree_contains_name(symbols, "Game"), "Game class not found in workspace symbols"
        assert SymbolUtils.symbol_tree_contains_name(symbols, "Player"), "Player class not found in workspace symbols"
        assert SymbolUtils.symbol_tree_contains_name(symbols, "MathUtils"), "MathUtils class not found in workspace symbols"

    @pytest.mark.parametrize("language_server", [Language.GDSCRIPT], indirect=True)
    def test_cross_file_references(self, language_server: SolidLanguageServer) -> None:
        file_path = os.path.join("src", "util", "math_utils.gd")
        doc_symbols = language_server.request_document_symbols(file_path)
        target_symbol = next((sym for sym in doc_symbols[0] if sym.get("name") == "calculate_level"), None)
        assert target_symbol is not None, "calculate_level function not discovered in math_utils.gd"
        start = target_symbol["selectionRange"]["start"]
        refs = language_server.request_references(file_path, start["line"], start["character"])
        assert any(
            ref.get("relativePath", "").replace("\\", "/").endswith("src/player.gd") for ref in refs
        ), "Player.level_up should reference MathUtils.calculate_level"

    @pytest.mark.parametrize("language_server", [Language.GDSCRIPT], indirect=True)
    def test_multiple_consumers_reference_power_level(self, language_server: SolidLanguageServer) -> None:
        file_path = os.path.join("src", "util", "math_utils.gd")
        doc_symbols = language_server.request_document_symbols(file_path)
        power_symbol = next((sym for sym in doc_symbols[0] if sym.get("name") == "calculate_power_level"), None)
        assert power_symbol is not None, "calculate_power_level function not discovered in math_utils.gd"
        start = power_symbol["selectionRange"]["start"]
        refs = language_server.request_references(file_path, start["line"], start["character"])
        referenced_files = {ref.get("relativePath", "").replace("\\", "/") for ref in refs}
        assert any(path.endswith("src/player.gd") for path in referenced_files), "Player should call calculate_power_level"
        assert any(path.endswith("src/game.gd") for path in referenced_files), "Game should call calculate_power_level"

    @pytest.mark.parametrize("language_server", [Language.GDSCRIPT], indirect=True)
    def test_document_symbols_for_player(self, language_server: SolidLanguageServer) -> None:
        file_path = os.path.join("src", "player.gd")
        doc_symbols = language_server.request_document_symbols(file_path)
        names = {sym.get("name") for sym in doc_symbols[0]}
        assert {"Player", "level_up", "_log_progress"}.issubset(names), "Expected Player symbols missing"
