import numpy as np
import pytest

from statetrace import FrozenReferenceError, Reference, read_frozen, schema, validate_transport


def test_freeze_without_validation_warns(fitted_small, tmp_path):
    with pytest.warns(UserWarning, match="without grouped validation"):
        p = fitted_small.freeze(tmp_path / "ref")
    assert p.name == "ref" + schema.FROZEN_SUFFIX and p.is_dir() and fitted_small.is_frozen


def test_roundtrip_directory_and_transport(validated_ref, transport_pair, tmp_path):
    _, qry, _ = transport_pair
    res = validated_ref.validation_
    p = validated_ref.freeze(tmp_path / "nvu.statetrace")
    assert not any(q.suffix in (".pkl", ".pickle") for q in p.rglob("*"))
    for npy in p.glob("*.npy"):
        np.load(npy, allow_pickle=False)
    b = read_frozen(p)
    assert b.manifest["format_version"] == schema.FORMAT_VERSION and b.manifest["validated"] is True
    assert b.manifest["keys"]["group"] == "donor_id"
    assert b.geometry.shape == (len(validated_ref.states_), 3)
    np.testing.assert_array_equal(b.latent, validated_ref.latent_)
    ref2 = Reference.load(p)
    assert ref2.is_frozen and not ref2.is_fitted and ref2.group_key == "donor_id"
    ref2.fit()
    assert ref2.validation_.overall["accuracy"] == pytest.approx(res.overall["accuracy"])
    assert ref2.distance_calibration_ is not None and len(ref2.distance_calibration_) == len(
        ref2.states_
    )
    np.testing.assert_allclose(ref2.geometry_, validated_ref.geometry_, rtol=1e-5)
    r_live = validate_transport(validated_ref, qry, group_key="donor_id")
    r_frozen = validate_transport(ref2, qry, group_key="donor_id")
    assert (r_live.states["status"] == r_frozen.states["status"]).all()
    assert ref2.leave_one_group_out().overall["accuracy"] == pytest.approx(res.overall["accuracy"])


def test_roundtrip_zip_and_guards(validated_ref, tmp_path):
    p = validated_ref.freeze(tmp_path / "ref.zip")
    assert p.is_file() and p.name == "ref.statetrace.zip"
    assert Reference.load(p).fit().is_fitted
    d = validated_ref.freeze(tmp_path / "ref")
    with pytest.raises(FileExistsError):
        validated_ref.freeze(tmp_path / "ref")
    validated_ref.freeze(tmp_path / "ref", overwrite=True)
    latent = d / "latent.npy"
    data = bytearray(latent.read_bytes())
    data[-1] ^= 0xFF
    latent.write_bytes(bytes(data))
    with pytest.raises(FrozenReferenceError, match="checksum"):
        read_frozen(d)
    read_frozen(d, verify=False)
    with pytest.raises(FrozenReferenceError):
        read_frozen(tmp_path / "nothing.statetrace")


def test_format_version_guards(validated_ref, tmp_path):
    p = validated_ref.freeze(tmp_path / "ref")
    m = p / schema.MANIFEST_FILE
    text = m.read_text()
    m.write_text(
        text.replace(f'"format_version": {schema.FORMAT_VERSION}', '"format_version": 999')
    )
    with pytest.raises(FrozenReferenceError, match="newer"):
        read_frozen(p)
    m.write_text(text.replace(f'"format_version": {schema.FORMAT_VERSION}', '"format_version": 1'))
    with pytest.raises(FrozenReferenceError, match="re-freeze"):
        read_frozen(p)
