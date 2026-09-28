from atlasroute import schema


def test_keys_are_unique_and_prefixed():
    names = [n for n in dir(schema) if n.startswith(("OBS_", "OBSM_", "UNS_"))]
    values = [getattr(schema, n) for n in names]
    assert len(set(values)) == len(values)
    for v in values:
        assert v.startswith("atlasroute") or v.startswith("X_atlasroute"), v


def test_constants():
    assert len(schema.STABILITY_CATEGORIES) == 4
    assert isinstance(schema.FORMAT_VERSION, int)
    assert schema.UNKNOWN_LABEL == "unknown"
    assert "leave_one_donor_out" in schema.VALIDATION_METHODS
