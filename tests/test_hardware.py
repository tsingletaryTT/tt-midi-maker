import json
from unittest.mock import patch, MagicMock
from tt_midi_maker.generation.hardware import detect_tt_devices, hardware_status


TT_SMI_OUTPUT = json.dumps({
    "device_info": [
        {"id": 0, "status": "available", "board_type": "N300", "arch": "wormhole"},
        {"id": 1, "status": "available", "board_type": "N300", "arch": "wormhole"},
    ]
})


def test_detects_devices_from_tt_smi():
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = TT_SMI_OUTPUT
    with patch("subprocess.run", return_value=mock_result):
        devices = detect_tt_devices()
    assert devices == [0, 1]


def test_returns_empty_when_tt_smi_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert detect_tt_devices() == []


def test_returns_empty_on_nonzero_exit():
    mock_result = MagicMock(returncode=1, stdout="")
    with patch("subprocess.run", return_value=mock_result):
        assert detect_tt_devices() == []


def test_returns_empty_on_invalid_json():
    mock_result = MagicMock(returncode=0, stdout="not json")
    with patch("subprocess.run", return_value=mock_result):
        assert detect_tt_devices() == []


def test_hardware_status_includes_device_count():
    mock_result = MagicMock(returncode=0, stdout=TT_SMI_OUTPUT)
    with patch("subprocess.run", return_value=mock_result):
        status = hardware_status()
    assert status["device_count"] == 2
    assert status["devices"][0]["board_type"] == "N300"


def test_hardware_status_no_hardware():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        status = hardware_status()
    assert status["device_count"] == 0
    assert status["tt_smi_available"] is False
