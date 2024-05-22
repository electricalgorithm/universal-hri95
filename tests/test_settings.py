"""
Test the settings module.
"""

from harmonicsradius.settings import SRAnalyzerSettings


def test_sr_analyzer_settings():
    """Test the default settings."""
    settings = SRAnalyzerSettings()
    assert settings.name == "SRAnalyzerDefault"
    assert settings.show_process is True
    assert settings.save_process is True
