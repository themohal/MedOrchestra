from tools.medical_research import _build_query


def test_build_query_empty():
    assert _build_query("", "") == ""


def test_build_query_includes_medical_context():
    query = _build_query("chest pain", "troponin high")
    assert "medical guideline" in query
    assert "chest pain" in query
    assert "troponin high" in query
