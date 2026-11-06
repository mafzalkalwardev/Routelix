# Piper voice models

Place a Piper voice model and its matching JSON metadata file in this directory.

Example:

```text
models/piper/en_US-lessac-medium.onnx
models/piper/en_US-lessac-medium.onnx.json
```

Voice model files are intentionally not committed to this public repository. Set
`TTS_ENGINE=piper` and `PIPER_MODEL_PATH` in `.env` after downloading a compatible
Piper voice model.
