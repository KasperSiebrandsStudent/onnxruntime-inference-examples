# Plugin TensorRT EP

This repo contains the plugin TRT EP implementation.

This plugin TRT EP is migrated from the original TRT EP and provides the implementations of `OrtEpFactory`, `OrtEp`, `OrtNodeComputeInfo`, `OrtDataTransferImpl` ... that are required for a plugin EP to be able to interact with ONNX Runtime via the EP ABI (introduced in ORT 1.23.0).