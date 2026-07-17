import pytest

from whisper_benchmark import resolve_mlx_model


def test_short_names_map_to_mlx_community_repos():
    assert resolve_mlx_model("tiny") == "mlx-community/whisper-tiny-mlx"
    assert resolve_mlx_model("base") == "mlx-community/whisper-base-mlx"
    assert resolve_mlx_model("small") == "mlx-community/whisper-small-mlx"
    assert resolve_mlx_model("medium") == "mlx-community/whisper-medium-mlx"
    assert resolve_mlx_model("large-v3") == "mlx-community/whisper-large-v3-mlx"
    assert resolve_mlx_model("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"


def test_full_repo_path_passes_through():
    assert resolve_mlx_model("someuser/custom-whisper") == "someuser/custom-whisper"


def test_unknown_short_name_raises_with_name_in_message():
    with pytest.raises(ValueError, match="large-v2"):
        resolve_mlx_model("large-v2")
