from onnx import helper
from qonnx.transformation.base import Transformation
from qonnx.custom_op.general.quant import quant


def quant_proto_to_quantized_tensor(model, node):
    inp_tensor = model.get_initializer(node.input[0])
    scale = model.get_initializer(node.input[1])
    zeropt = model.get_initializer(node.input[2])
    bitwidth = model.get_initializer(node.input[3])
    signed = helper.get_node_attr_value(node, 'signed')
    narrow = helper.get_node_attr_value(node, 'narrow')
    rounding_mode = helper.get_node_attr_value(node, 'rounding_mode').decode('ascii')
    quant_tensor = quant(inp_tensor, scale, zeropt, bitwidth, signed, narrow, rounding_mode)
    return quant_tensor


class QuantizedBiasToInitializer(Transformation):
    """
       If a quantized bias is found it is transformed into a constant initializer using the quant function.
    """
    def apply(self, model):
        graph = model.graph
        for node in graph.node:
            if node.op_type in ("Conv") and len(node.input) > 2:
                producer = model.find_producer(node.input[2])
                if producer is None or producer.op_type != 'Quant':
                    break
                quant_bias = quant_proto_to_quantized_tensor(model, producer)
                init_name = model.make_new_valueinfo_name()
                model.set_initializer(init_name, quant_bias)
                graph.node.remove(producer)
                node.input[2] = init_name 
                return model, True

        return model, False
