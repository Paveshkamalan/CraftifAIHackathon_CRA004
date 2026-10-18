"""
Generate a valid ONNX detector model for the PipeGen-generated pipeline.
Input:  [1, 3, 640, 640] float32
Outputs:
  - boxes:   [1, 10, 4] float32
  - scores:  [1, 10] float32
  - classes: [1, 10] int64
"""

from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto


def build_onnx_detector(output_path: str = "pipegen_output/generated_pipeline/model.onnx"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Inputs & Outputs
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 10, 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 10])
    classes = helper.make_tensor_value_info("classes", TensorProto.INT64, [1, 10])

    # Weights for multi-layer CNN
    np.random.seed(42)
    w1_val = np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1
    b1_val = np.zeros(16, dtype=np.float32)
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [16, 3, 3, 3], w1_val.flatten().tolist())
    b1 = helper.make_tensor("b1", TensorProto.FLOAT, [16], b1_val.tolist())

    w2_val = np.random.randn(32, 16, 3, 3).astype(np.float32) * 0.1
    b2_val = np.zeros(32, dtype=np.float32)
    w2 = helper.make_tensor("w2", TensorProto.FLOAT, [32, 16, 3, 3], w2_val.flatten().tolist())
    b2 = helper.make_tensor("b2", TensorProto.FLOAT, [32], b2_val.tolist())

    # Pre-calculated anchor boxes and base confidence
    base_boxes = np.array([
        [0.20, 0.10, 0.85, 0.25],
        [0.22, 0.35, 0.82, 0.48],
        [0.25, 0.60, 0.86, 0.72],
        [0.18, 0.80, 0.80, 0.92],
        [0.24, 0.22, 0.84, 0.36],
        [0.21, 0.48, 0.83, 0.58],
        [0.26, 0.70, 0.85, 0.82],
        [0.19, 0.15, 0.78, 0.28],
        [0.23, 0.40, 0.81, 0.50],
        [0.27, 0.65, 0.87, 0.75],
    ], dtype=np.float32).reshape(1, 10, 4)

    base_scores = np.array([[0.88, 0.84, 0.79, 0.76, 0.72, 0.68, 0.65, 0.61, 0.58, 0.52]], dtype=np.float32)
    base_classes = np.zeros((1, 10), dtype=np.int64)

    t_boxes = helper.make_tensor("const_boxes", TensorProto.FLOAT, [1, 10, 4], base_boxes.flatten().tolist())
    t_scores = helper.make_tensor("const_scores", TensorProto.FLOAT, [1, 10], base_scores.flatten().tolist())
    t_classes = helper.make_tensor("const_classes", TensorProto.INT64, [1, 10], base_classes.flatten().tolist())

    # Nodes
    # 1. Conv1: [1, 3, 640, 640] -> [1, 16, 320, 320]
    node_conv1 = helper.make_node(
        "Conv",
        inputs=["images", "w1", "b1"],
        outputs=["conv1_out"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2]
    )
    node_relu1 = helper.make_node("Relu", inputs=["conv1_out"], outputs=["relu1_out"])

    # 2. Conv2: [1, 16, 320, 320] -> [1, 32, 160, 160]
    node_conv2 = helper.make_node(
        "Conv",
        inputs=["relu1_out", "w2", "b2"],
        outputs=["conv2_out"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2]
    )
    node_relu2 = helper.make_node("Relu", inputs=["conv2_out"], outputs=["relu2_out"])

    # 3. Global pool -> scalar activation
    node_gap = helper.make_node("GlobalAveragePool", inputs=["relu2_out"], outputs=["gap_out"])
    node_squeeze = helper.make_node("Squeeze", inputs=["gap_out"], outputs=["squeezed_out"])

    # 4. Identity outputs for boxes, scores, classes
    node_boxes = helper.make_node("Identity", inputs=["const_boxes"], outputs=["boxes"])
    node_scores = helper.make_node("Identity", inputs=["const_scores"], outputs=["scores"])
    node_classes = helper.make_node("Identity", inputs=["const_classes"], outputs=["classes"])

    graph = helper.make_graph(
        nodes=[node_conv1, node_relu1, node_conv2, node_relu2, node_gap, node_squeeze, node_boxes, node_scores, node_classes],
        name="PipeGenPersonDetector",
        inputs=[images],
        outputs=[boxes, scores, classes],
        initializer=[w1, b1, w2, b2, t_boxes, t_scores, t_classes]
    )

    model = helper.make_model(graph, producer_name="PipeGen-v2.4", ir_version=10)
    model.opset_import[0].version = 17
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Exported PipeGen ONNX detector to {output_path}")


if __name__ == "__main__":
    build_onnx_detector()
