"""Keep synthetic research accounting separate from the project's real history."""
import pytest


@pytest.fixture(autouse=True)
def isolated_project_research_registry(tmp_path, monkeypatch):
    from rles_orchestrator import research_access
    from rles_research.access import ProjectResearchRegistry

    monkeypatch.setattr(research_access, "project_registry",
                        lambda: ProjectResearchRegistry(tmp_path / "project-research.sqlite"))
