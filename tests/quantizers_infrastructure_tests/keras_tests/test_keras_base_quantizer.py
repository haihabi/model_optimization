# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from model_compression_toolkit import QuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig, \
    NodeActivationQuantizationConfig
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs
from tests.quantizers_infrastructure_tests.keras_tests.base_keras_infrastructure_test import \
    BaseKerasInfrastructureTest, ZeroWeightsQuantizer, ZeroActivationsQuantizer, dummy_fn


class TestKerasBaseWeightsQuantizer(BaseKerasInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_weights_quantization_config(self):
        op_cfg, _ = get_op_quantization_configs()
        qc = QuantizationConfig()
        op_cfg_uniform = op_cfg.clone_and_edit(activation_quantization_method=QuantizationMethod.UNIFORM,
                                               weights_quantization_method=QuantizationMethod.UNIFORM)
        return NodeWeightsQuantizationConfig(qc, op_cfg_uniform, dummy_fn, dummy_fn, -1)

    def run_test(self):
        with self.unit_test.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(self.get_weights_quantization_config())
        self.unit_test.assertEqual(f'Quantization method mismatch expected: [<QuantizationMethod.POWER_OF_TWO: 0>, '
                                   f''f'<QuantizationMethod.SYMMETRIC: 3>] and got  QuantizationMethod.UNIFORM',
                                   str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(self.get_activation_quantization_config())
        self.unit_test.assertEqual(f'Expect weight quantization got activation', str(e.exception))

        weight_quantization_config = super(TestKerasBaseWeightsQuantizer, self).get_weights_quantization_config()
        quantizer = ZeroWeightsQuantizer(weight_quantization_config)
        self.unit_test.assertTrue(quantizer.quantization_config == weight_quantization_config)


class TestKerasBaseActivationsQuantizer(BaseKerasInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_activation_quantization_config(self):
        op_cfg, _ = get_op_quantization_configs()
        qc = QuantizationConfig()
        op_cfg_uniform = op_cfg.clone_and_edit(activation_quantization_method=QuantizationMethod.UNIFORM,
                                               weights_quantization_method=QuantizationMethod.UNIFORM)
        return NodeActivationQuantizationConfig(qc, op_cfg_uniform, dummy_fn, dummy_fn)

    def run_test(self):
        with self.unit_test.assertRaises(Exception) as e:
            ZeroActivationsQuantizer(self.get_activation_quantization_config())
        self.unit_test.assertEqual(f'Quantization method mismatch expected: [<QuantizationMethod.POWER_OF_TWO: 0>, '
                                   f'<QuantizationMethod.SYMMETRIC: 3>] and got  QuantizationMethod.UNIFORM',
                                   str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ZeroActivationsQuantizer(self.get_weights_quantization_config())
        self.unit_test.assertEqual(f'Expect activation quantization got weight', str(e.exception))

        activation_quantization_config = super(TestKerasBaseActivationsQuantizer,
                                               self).get_activation_quantization_config()
        quantizer = ZeroActivationsQuantizer(activation_quantization_config)
        self.unit_test.assertTrue(quantizer.quantization_config == activation_quantization_config)