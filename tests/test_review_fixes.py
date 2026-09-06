from __future__ import annotations

import pytest

from core import config
from core.config import as_bool
from core.errors import InvalidRequest


class TestEmptyFormFields:
    def test_empty_string_falls_back_to_the_default(self):
        assert as_bool("", True) is True
        assert as_bool("   ", True) is True
        assert as_bool("", False) is False

    def test_explicit_values_still_win(self):
        assert as_bool("false", True) is False
        assert as_bool("true", False) is True
        assert as_bool("on", False) is True
        assert as_bool(None, True) is True

    def test_a_nonsense_value_is_rejected_rather_than_guessed(self):
        with pytest.raises(InvalidRequest):
            as_bool("maybe", True)

    def test_the_n8n_empty_optional_flag_path_works(self, all_installed):
        """n8n cannot omit a form field: it sends every parameter it declares,
        empty when the operator left it alone. An empty flag must read as
        absent, not as false."""
        built = config.build({
            "target_language": "vi",
            "tts_model": "edge",
            "enable_diarization": "false",
            "enable_alignment": "",
            "enable_source_separation": "false",
        })
        assert built.features.alignment is True
        assert built.features.diarization is True

    def test_an_empty_alignment_flag_keeps_the_default(self, all_installed):
        built = config.build({"target_language": "vi", "enable_alignment": ""})
        assert built.features.alignment is True


class TestAlignmentPlanLength:
    def test_one_plan_per_segment_even_when_nothing_was_measured(self):
        from dubflow_core import alignment

        segments = [
            {"start": 0.0, "end": 2.0},
            {"start": 2.0, "end": 4.0, "tts_duration_raw": 4.0},
            {"start": 4.0, "end": 6.0},
        ]
        plans = alignment.plan_segments(segments, media_duration=10.0)
        assert len(plans) == len(segments), "plans would zip onto the wrong lines"
        assert plans[0].status == "unmeasured"
        assert plans[1].status in {"aligned", "clamped"}


class TestSharedMixing:
    def test_the_two_services_use_the_same_gains(self):
        from dubflow_core import mixing

        voiceover = mixing.blend_filtergraph(separated=False)
        separated = mixing.blend_filtergraph(separated=True)
        assert "volume=0.25" in voiceover and "volume=1.5" in voiceover
        assert "volume=1.0" in separated
        assert voiceover.count("aresample=48000") == 2

    def test_atempo_is_chained_past_the_filter_limit(self):
        from dubflow_core import mixing

        assert mixing.atempo_filter(1.25) == "atempo=1.2500"
        assert mixing.atempo_filter(3.0) == "atempo=2.0,atempo=1.5000"
        assert mixing.atempo_filter(0.3) == "atempo=0.5,atempo=0.6000"

    def test_gateway_does_not_build_its_own_graph(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        source = (root / "ai-service/app.py").read_text(encoding="utf-8")
        assert "loudnorm" not in source
        assert "adelay=" not in source
        assert "atempo=" not in source


class TestOutdatedBackend:
    def service(self, monkeypatch, status=404, health=None):  # noqa: ANN201
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai-service"))
        import app

        class Response:
            status_code = status
            text = "Not Found"

            def json(self):  # noqa: ANN201
                return {"detail": "Not Found"}

        monkeypatch.setattr(app, "_resolve_backend",
                            lambda force=False: ("colab", "https://x.example", ""))
        monkeypatch.setattr(app.httpx, "request", lambda *a, **k: Response())
        monkeypatch.setattr(app, "_backend_version", lambda url, token: health)
        monkeypatch.setattr(app, "_capabilities_cache", None)
        return app

    def test_every_new_endpoint_reports_the_version_and_the_restart(self, monkeypatch):
        app = self.service(monkeypatch, health="3.0")
        for path in app.MODERN_ENDPOINTS:
            with pytest.raises(app.OutdatedBackend) as failure:
                app._colab_request(path, json={})
            message = str(failure.value.detail)
            assert "running AI service 3.0" in message
            assert path in message
            assert "Restart session" in message
            assert "make colab" in message

    def test_a_session_reporting_no_version_is_described_honestly(self, monkeypatch):
        app = self.service(monkeypatch, health=None)
        with pytest.raises(app.OutdatedBackend) as failure:
            app._colab_request("/validate", json={})
        assert "a build older than 5.0" in str(failure.value.detail)

    def test_capabilities_labels_it_outdated_rather_than_offline(self, monkeypatch):
        app = self.service(monkeypatch, health="3.0")
        payload = app.capabilities()
        assert payload["available"] is False
        assert payload["reason"] == "outdated"
        assert "Restart session" in payload["hint"]
        assert payload["languages"] and payload["stages"]

    def test_an_ordinary_error_is_not_blamed_on_the_version(self, monkeypatch):
        app = self.service(monkeypatch, status=500)
        with pytest.raises(Exception) as failure:
            app._colab_request("/validate", json={})
        assert not isinstance(failure.value, app.OutdatedBackend)
