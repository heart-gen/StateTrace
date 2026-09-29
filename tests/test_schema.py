from statetrace import schema


def test_keys_are_unique_and_prefixed():
    names = [n for n in dir(schema) if n.startswith(("OBS_", "OBSM_", "UNS_"))]
    values = [getattr(schema, n) for n in names]
    assert len(set(values)) == len(values)
    for v in values:
        assert v.startswith("statetrace") or v.startswith("X_statetrace"), v


def test_vocabularies():
    assert schema.STABILITY_CATEGORIES == ("stable", "moderate", "fragile", "insufficient")
    assert "supported_with_caution" in schema.TRANSPORT_STATUS
    assert schema.CELL_CATEGORIES == ("supported", "uncertain", "out_of_reference")
    assert schema.FORMAT_VERSION == 3 and schema.FROZEN_SUFFIX == ".statetrace"
