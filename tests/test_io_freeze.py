import numpy as np
import pytest

from atlasroute import FrozenReferenceError, Reference, read_frozen, schema


def test_freeze_without_validation_warns(fitted_small, tmp_path):
    with pytest.warns(UserWarning, match="without leave-one-donor-out"):
        p = fitted_small.freeze(tmp_path / "ref")
    assert p.name == "ref" + schema.FROZEN_SUFFIX and p.is_dir()
    assert fitted_small.is_frozen


def test_roundtrip_directory(fitted_small, tmp_path):
    res = fitted_small.leave_one_donor_out()
    p = fitted_small.freeze(tmp_path / "nvu.atlasroute")
    assert not any(q.suffix in (".pkl", ".pickle") for q in p.rglob("*"))
    for npy in p.glob("*.npy"):
        np.load(npy, allow_pickle=False)
    b = read_frozen(p)
    assert b.manifest["format_version"] == schema.FORMAT_VERSION
    assert b.manifest["validated"] is True
    assert b.manifest["keys"]["donor"] == "donor_id"
    np.testing.assert_array_equal(b.latent, fitted_small.latent_)
    assert b.states == list(fitted_small.states_) and b.donors == list(fitted_small.donors_)
    assert b.celltypes == list(fitted_small.celltypes_)
    assert b.validation.overall == res.overall

    ref2 = Reference.load(p)
    assert ref2.is_frozen and not ref2.is_fitted
    ref2.fit()
    assert ref2.is_fitted
    assert list(ref2.state_table().index) == list(fitted_small.states_)
    assert ref2.validation_.overall["accuracy"] == pytest.approx(res.overall["accuracy"])
    res2 = ref2.leave_one_donor_out()
    assert res2.overall["accuracy"] == pytest.approx(res.overall["accuracy"])
    assert ref2.reference_id_ == fitted_small.reference_id_


def test_roundtrip_zip(fitted_small, tmp_path):
    fitted_small.leave_one_donor_out()
    p = fitted_small.freeze(tmp_path / "ref.zip")
    assert p.is_file() and p.name == "ref.atlasroute.zip"
    ref2 = Reference.load(p).fit()
    assert ref2.leave_one_donor_out().overall["n_cells"] == fitted_small.latent_.shape[0]


def test_overwrite_guard(fitted_small, tmp_path):
    fitted_small.leave_one_donor_out()
    p = fitted_small.freeze(tmp_path / "ref")
    with pytest.raises(FileExistsError):
        fitted_small.freeze(tmp_path / "ref")
    fitted_small.freeze(tmp_path / "ref", overwrite=True)
    assert p.is_dir()


def test_tamper_detection(fitted_small, tmp_path):
    fitted_small.leave_one_donor_out()
    p = fitted_small.freeze(tmp_path / "ref")
    latent = p / "latent.npy"
    data = bytearray(latent.read_bytes())
    data[-1] ^= 0xFF
    latent.write_bytes(bytes(data))
    with pytest.raises(FrozenReferenceError, match="checksum"):
        read_frozen(p)
    read_frozen(p, verify=False)


def test_missing_and_newer_format(tmp_path, fitted_small):
    with pytest.raises(FrozenReferenceError):
        read_frozen(tmp_path / "nothing.atlasroute")
    fitted_small.leave_one_donor_out()
    p = fitted_small.freeze(tmp_path / "ref")
    m = p / schema.MANIFEST_FILE
    m.write_text(
        m.read_text().replace(f'"format_version": {schema.FORMAT_VERSION}', '"format_version": 999')
    )
    with pytest.raises(FrozenReferenceError, match="newer"):
        read_frozen(p)
