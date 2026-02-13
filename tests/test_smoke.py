def test_imports():
    import importlib.util
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    init_path = project_root / "src" / "news_reporter" / "__init__.py"

    spec = importlib.util.spec_from_file_location("news_reporter", init_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    assert module is not None
