def test_pandas_not_imported():
    import sys
    assert "pandas" not in sys.modules, "pandas should not be imported anywhere in HPLC"