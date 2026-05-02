from pathlib import Path

from scripts import run_llm_clue_fixture_experiments as fixtures


def test_run_fixture_experiments_writes_summary(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "01_first.yaml").write_text("name: first\n", encoding="utf-8")
    (config_dir / "02_second.yaml").write_text("name: second\n", encoding="utf-8")
    calls: list[tuple[Path, Path, int]] = []

    def fake_probe(**kwargs):
        calls.append((kwargs["config_path"], kwargs["out_dir"], kwargs["max_packets"]))
        return {"manifest": str(kwargs["out_dir"] / "manifest.json")}

    monkeypatch.setattr(fixtures, "run_llm_counterfactual_clue_probe", fake_probe)

    summary = fixtures.run_fixture_experiments(
        config_dir=config_dir,
        out_dir=tmp_path / "out",
        split="train",
        card_top_k=8,
        max_packets=16,
        llm_backend="mock",
        execute_tests=True,
        test_split="test",
    )

    assert [call[0].name for call in calls] == ["01_first.yaml", "02_second.yaml"]
    assert [call[1].name for call in calls] == ["01_first", "02_second"]
    assert {call[2] for call in calls} == {16}
    assert summary["max_packets"] == 16
    assert (tmp_path / "out" / "manifest.json").exists()