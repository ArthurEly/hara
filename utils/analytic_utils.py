import logging
import math
import numpy as np
import onnx
from onnx import helper
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Importa o toolkit 3D de forma segura
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    Axes3D = None  # Define como None se não for encontrado

# Configura um logger específico para este módulo, o que é uma boa prática
logger = logging.getLogger(__name__)


# Mapeamento global de regras de folding
FOLDING_RULES = {
    "Addstreams": {"parameter": "PE", "constraint": "inp_channels"},
    "ChannelwiseOp": {"parameter": "PE", "constraint": "channels"},
    "ConvolutionInputGenerator": {"parameter": "SIMD", "constraint": "IFMChannels"},
    "Downsampler": {"parameter": "SIMD", "constraint": "inp_channels"},
    "DuplicateStreams": {"parameter": "PE", "constraint": "channels"},
    "StreamingEltwise": {"parameter": "PE", "constraint": "inp_channels"},
    "FMPadding": {"parameter": "SIMD", "constraint": "NumChannels"},
    "FMPadding_Pixel": {"parameter": "SIMD", "constraint": "NumChannels"},
    "Globalaccpool": {"parameter": "PE", "constraint": "channels"},
    "LabelSelect": {"parameter": "PE", "constraint": "Labels"},
    "MVAU": {"parameter": ["PE", "SIMD"], "constraint": ["MH", "MW"]},
    "MatrixVectorActivation": {"parameter": ["PE", "SIMD"], "constraint": ["MH", "MW"]},
    "Pool": {"parameter": "PE", "constraint": "NumChannels"},
    "Thresholding": {"parameter": "PE", "constraint": "MH"},
    "VectorVectorActivation": {"parameter": ["PE", "SIMD"], "constraint": ["k_h * k_w", "channels"]},
}


class FinnCycleEstimator:
    """
    Estima a latência em ciclos de um modelo FINN em formato ONNX,
    gerando análises visuais do impacto do paralelismo (PE e SIMD).
    """

    def __init__(self, onnx_path: str, debug: bool = False):
        """
        Inicializa o analisador.

        Args:
            onnx_path (str): Caminho para o arquivo .onnx do modelo.
            debug (bool): Se True, ativa as mensagens de depuração detalhadas.
        """
        self.logger = logger
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        if not onnx_path:
            raise ValueError("O caminho para o arquivo ONNX é necessário.")

        self.model = onnx.load(onnx_path)
        self.graph = self.model.graph

        # Mapeia as formas de todos os tensores para fácil acesso
        self.tensor_shapes = {
            tensor.name: [d.dim_value for d in tensor.type.tensor_type.shape.dim]
            for tensor in list(self.graph.value_info) + list(self.graph.input) + list(self.graph.output)
        }
        self.logger.info(f"Modelo {onnx_path} carregado com {len(self.graph.node)} nós.")

    # --- Seção: Métodos Auxiliares para Análise do Grafo ---

    def _get_node_attributes(self, node: onnx.NodeProto) -> dict:
        """Extrai os atributos de um nó para um dicionário."""
        return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}

    def _get_tensor_shape(self, tensor_name: str) -> list:
        """Obtém a forma de um tensor pelo nome."""
        return self.tensor_shapes.get(tensor_name, [])

    def _find_downstream_node(self, current_node_output_tensor: str) -> onnx.NodeProto | None:
        """Encontra o nó que consome um determinado tensor de saída."""
        for node in self.graph.node:
            if current_node_output_tensor in node.input:
                return node
        return None

    def _infer_kernel_dim_from_downstream_mvau(self, node: onnx.NodeProto, ifm_shape: list) -> list | None:
        """
        Tenta inferir o KernelDim a partir do nó MVAU seguinte,
        assumindo um kernel quadrado.
        """
        self.logger.debug("  [3] 'KernelDim' não encontrado. Tentando inferir a partir do nó seguinte...")
        downstream_node = self._find_downstream_node(node.output[0])

        if not (downstream_node and downstream_node.op_type.startswith("MVAU")):
            self.logger.debug("  [✗] FALHA: Não foi encontrado um nó MVAU válido a seguir.")
            return None

        self.logger.debug(f"  [4] Nó seguinte encontrado: {downstream_node.name} ({downstream_node.op_type})")
        mw = self._get_node_attributes(downstream_node).get("MW")
        self.logger.debug(f"  [5] Atributo 'MW' do nó seguinte: {mw}")
        ifm_ch = ifm_shape[1]

        if not (mw is not None and ifm_ch > 0):
            self.logger.debug("  [✗] FALHA: Atributo 'MW' ou canais de entrada inválidos para inferência.")
            return None

        kernel_area = mw / ifm_ch
        self.logger.debug(f"  [6] Cálculo da área do kernel: {mw} (MW) / {ifm_ch} (Canais) = {kernel_area}")
        k = int(round(math.sqrt(kernel_area)))

        if k * k == kernel_area:
            self.logger.debug(f"  [✓] SUCESSO: Kernel inferido como {k}x{k}")
            return [k, k]
        
        self.logger.debug(f"  [✗] FALHA: A área do kernel ({kernel_area}) não é um quadrado perfeito.")
        return None

    def _calculate_conv_common_params(self, node: onnx.NodeProto) -> dict | None:
        """Extrai parâmetros comuns de camadas convolucionais."""
        self.logger.debug(f"\n--- Depurando o nó: {node.name} ---")
        attrs = self._get_node_attributes(node)
        ifm_shape = self._get_tensor_shape(node.input[0])
        ofm_shape = self._get_tensor_shape(node.output[0])
        self.logger.debug(f"  [1] Formas dos tensores: Entrada={ifm_shape}, Saída={ofm_shape}")

        if len(ifm_shape) < 4 or len(ofm_shape) < 4:
            self.logger.debug("  [✗] FALHA: Formas de tensor inválidas.")
            return None

        kernel_dim = attrs.get("ConvKernelDim")
        self.logger.debug(f"  [2] 'KernelDim' encontrado diretamente: {kernel_dim}")
        if kernel_dim is None:
            kernel_dim = self._infer_kernel_dim_from_downstream_mvau(node, ifm_shape)

        if kernel_dim is None:
            self.logger.debug(f"--- Fim da depuração para {node.name}: Não foi possível determinar KernelDim ---")
            return None
        
        self.logger.debug(f"--- Fim da depuração para {node.name}: Parâmetros extraídos com sucesso ---")
        stride_dim = attrs.get("Stride", [1, 1])
        dilation_dim = attrs.get("Dilation", [1, 1])
        
        self.logger.debug({
            "ifm_ch": ifm_shape[3], "ifm_dim_h": ifm_shape[1], "ifm_dim_w": ifm_shape[2],
            "ofm_dim_h": ofm_shape[1], "ofm_dim_w": ofm_shape[2],
            "k_h": kernel_dim[0], "k_w": kernel_dim[1], "stride_w": stride_dim[1],
            "dilation_h": dilation_dim[0], "dilation_w": dilation_dim[1],
        })
        
        return {
            "ifm_ch": ifm_shape[3], "ifm_dim_h": ifm_shape[1], "ifm_dim_w": ifm_shape[2],
            "ofm_dim_h": ofm_shape[1], "ofm_dim_w": ofm_shape[2],
            "k_h": kernel_dim[0], "k_w": kernel_dim[1], "stride_w": stride_dim[1],
            "dilation_h": dilation_dim[0], "dilation_w": dilation_dim[1],
        }

    # --- Seção: Funções de Cálculo de Ciclos por Camada ---

    def _calculate_convinputgen_hls_cycles(self, node: onnx.NodeProto) -> dict | None:
        params = self._calculate_conv_common_params(node)
        if not params: return None
        
        attrs = self._get_node_attributes(node)
        parallel_window = attrs.get("parallel_window", 0)

        if parallel_window:
            num_input_elems = params["ifm_dim_h"] * params["ifm_dim_w"] * params["ifm_ch"]
            total_val = num_input_elems + 2
        else:
            write_block = params["ofm_dim_w"] * params["k_w"] * params["k_h"] * params["ifm_ch"]
            read_block = params["stride_w"] * params["ifm_dim_w"] * params["ifm_ch"]
            buffer_fill = params["ifm_dim_w"] * params["k_h"] * params["dilation_h"] * params["ifm_ch"]
            
            processing_cycles = params['ofm_dim_h'] * max(write_block, read_block)
            total_val = buffer_fill + processing_cycles
        
        return {"formula": f"{total_val}/SIMD", "IFMChannels": params["ifm_ch"], "parallel_window": parallel_window}

    def _calculate_convinputgen_rtl_cycles(self, node: onnx.NodeProto) -> dict | None:
        self.logger.debug(f"\n--- _calculate_convinputgen_rtl_cycles ---")
        
        params = self._calculate_conv_common_params(node)
        if not params: return None
        
        self.logger.debug(params)
        
        attrs = self._get_node_attributes(node)
        parallel_window = attrs.get("parallel_window", 0)
        self.logger.debug(f"parallel_window: {parallel_window}")

        if parallel_window:
            num_input_elems = params["ifm_dim_h"] * params["ifm_dim_w"] * params["ifm_ch"]
            total_val = num_input_elems + 2
        else:
            # Tamanho do buffer necessário para iniciar a primeira convolução
            lines_to_buffer = (params["k_h"] - 1) * params["dilation_h"] * params["ifm_dim_w"]
            pixels_in_last_line = (params["k_w"] - 1) * params["dilation_w"] + 1
            buffer_min_size = (lines_to_buffer + pixels_in_last_line) * params["ifm_ch"]
            
            write_block = params["ofm_dim_w"] * params["k_w"] * params["k_h"] * params["ifm_ch"]
            read_block = params["stride_w"] * params["ifm_dim_w"] * params["ifm_ch"]
            
            processing_cycles = params['ofm_dim_h'] * max(write_block, read_block)
            total_val = buffer_min_size + processing_cycles
        
        self.logger.debug({"formula": f"{total_val}/SIMD", "IFMChannels": params["ifm_ch"], "parallel_window": parallel_window})
        return {"formula": f"{total_val}/SIMD", "IFMChannels": params["ifm_ch"], "parallel_window": parallel_window}

    def _calculate_mvau_cycles(self, node: onnx.NodeProto) -> dict | None:
        attrs = self._get_node_attributes(node)
        mh, mw = attrs.get("MH"), attrs.get("MW")
        
        # DEBUG: Imprime os valores extraídos
        self.logger.debug(f"DEBUG: MVAU node '{node.name}' - MH: {mh}, MW: {mw}")

        if mh is None or mw is None: return None
        
        numInputVectors = attrs.get("numInputVectors")
        # np.prod(num_inp_vec) é o produto das dimensões espaciais (ex: 30x30 = 900)
        self.logger.debug(f"DEBUG: MVAU node '{node.name}' - numInputVectors shape: {attrs.get('numInputVectors')}")
        num_inp_vec_prod = np.prod(numInputVectors) if len(numInputVectors) > 1 else 1
        
        self.logger.debug(f"DEBUG: MVAU node '{node.name}' - numInputVectors: {num_inp_vec_prod}")
        
        return {"formula": f"({mh}/PE) * ({mw}/SIMD) * {num_inp_vec_prod}", "MH": mh, "MW": mw, "numInputVectors": num_inp_vec_prod}

    def _calculate_thresholding_cycles(self, node: onnx.NodeProto) -> dict | None:
        ifm_shape = self._get_tensor_shape(node.input[0])
        if len(ifm_shape) == 2:
            _, channels = ifm_shape
            fmdim_h, fmdim_w = 1, 1
        elif len(ifm_shape) >= 4:
            _, channels, fmdim_h, fmdim_w = ifm_shape[:4]
        else:
            return None
        
        total_elements = channels * 1 * fmdim_h * fmdim_w # batch_size = 1
        return {"formula": f"{total_elements}/PE", "MH": channels}

    def _calculate_labelselect_cycles(self, node: onnx.NodeProto) -> dict | None:
        attrs = self._get_node_attributes(node)
        nlabels = attrs.get("Labels")
        if nlabels is None: return None
        
        return {"formula": f"{nlabels}/PE", "Labels": nlabels}

    def _calculate_streamingmaxpool_cycles(self, node: onnx.NodeProto) -> dict | None:
        attrs = self._get_node_attributes(node)
        ifm_dim = attrs.get("ImgDim")
        pool_dim = attrs.get("PoolDim")
        if ifm_dim is None or pool_dim is None or len(ifm_dim) < 2 or len(pool_dim) < 2: return None
        
        # Fórmula: ifm_dim[1]*ifm_dim[1]*(1+1/(pool_dim[1]*pool_dim[1]))
        # Nota: Assume dimensões e pooling quadrados
        dim = ifm_dim[1]
        pool = pool_dim[1]
        exp_cycles = int(dim * dim * (1 + 1 / (pool * pool)))
        
        return {"formula": f"{exp_cycles}"}

    def _calculate_fmpadding_cycles(self, node: onnx.NodeProto) -> dict | None:
        ifm_shape = self._get_tensor_shape(node.input[0])
        ofm_shape = self._get_tensor_shape(node.output[0])

        if len(ifm_shape) < 4 or len(ofm_shape) < 4: return None
        
        channels = ifm_shape[3]
        _, odim_h, odim_w, _ = ofm_shape
        total_elements = channels * 1 * odim_h * odim_w # batch_size = 1
        
        return {"formula": f"{total_elements}/SIMD", "NumChannels": channels}

    # --- Seção: Funções de Análise do Grafo ---

    def get_cycle_formulas(self, op_type_filter: str | None = None) -> dict:
        """
        Itera sobre o grafo e retorna um dicionário com as fórmulas de
        ciclos para as camadas de processamento suportadas, com filtro.
        """
        formulas = {}
        unsupported_nodes = []
        ignored_ops = {"Constant", "StreamingFIFO_rtl", "StreamingDataWidthConverter_rtl", "StreamingDataWidthConverter_hls"}
        
        op_to_function_map = {
            "ConvolutionInputGenerator": self._calculate_convinputgen_rtl_cycles,
            "MVAU": self._calculate_mvau_cycles,
            "MatrixVectorActivation": self._calculate_mvau_cycles,
            "Thresholding": self._calculate_thresholding_cycles,
            "LabelSelect": self._calculate_labelselect_cycles,
            "StreamingMaxPool": self._calculate_streamingmaxpool_cycles,
            "FMPadding": self._calculate_fmpadding_cycles,
        }

        for node in self.graph.node:
            op_type = node.op_type
            self.logger.debug(f"Analisando nó: {node.name} ({op_type})")
            
            # Aplica o filtro se especificado
            if op_type_filter and op_type_filter not in op_type:
                continue

            calculation_func = None
            for key, func in op_to_function_map.items():
                if key in op_type:
                    if key == "ConvolutionInputGenerator":
                        if "_hls" in op_type:
                            calculation_func = self._calculate_convinputgen_hls_cycles
                        else:
                            calculation_func = self._calculate_convinputgen_rtl_cycles
                    else:
                        calculation_func = func
                    break
            
            if calculation_func:
                self.logger.debug(f"  Função de cálculo encontrada para {op_type}: {calculation_func.__name__}")
                result = calculation_func(node)
                self.logger.debug(f"{result}")
                if result:
                    formulas[node.name] = {"op_type": op_type, **result}
                elif op_type not in ignored_ops:
                    unsupported_nodes.append(f"{node.name} ({op_type})")
            elif op_type not in ignored_ops:
                unsupported_nodes.append(f"{node.name} ({op_type})")

        if unsupported_nodes:
            self.logger.warning("\\nAVISO: Os nós a seguir não puderam ser analisados: %s", ", ".join(unsupported_nodes))
        
        self.logger.debug(f"Fórmulas extraídas: {formulas}")
        return formulas
    
    # --- Seção: Funções de Análise e Plotagem ---
    
    def _eval_formula(self, formula: str, eval_params: dict) -> float:
        """Avalia uma fórmula de string com parâmetros dados, lidando com erros."""
        eval_scope = {"__builtins__": None, "math": math}
        try:
            result = eval(formula, eval_scope, eval_params)
            return result if result > 0 else np.inf
        except ZeroDivisionError:
            return np.inf
        except Exception as e:
            self.logger.error(f"Erro ao avaliar a fórmula '{formula}' com params {eval_params}: {e}")
            return np.inf

    def _find_next_valid_parallelism(self, layer_name: str, current_value: int, op_type: str, layer_params: dict, param_to_update: str) -> int:
        """Encontra o próximo valor de paralelismo válido (PE ou SIMD)."""
        rule = None
        for key, value in FOLDING_RULES.items():
            if key in op_type:
                rule = value
                break
        
        if not rule or "constraint" not in rule:
            self.logger.warning(f"Regra de folding não encontrada ou incompleta para {op_type}. Retornando o teto do valor ideal.")
            return current_value + 1

        # Determina o parâmetro de restrição com base no parâmetro que está sendo atualizado (PE ou SIMD)
        constraint_param_value = None
        
        if isinstance(rule["parameter"], list):
            if param_to_update == rule["parameter"][0]:
                constraint_param = rule["constraint"][0]
                constraint_param_value = layer_params.get(constraint_param)
            elif param_to_update == rule["parameter"][1]:
                constraint_param = rule["constraint"][1]
                constraint_param_value = layer_params.get(constraint_param)
        else:
            constraint_param = rule["constraint"]
            constraint_param_value = layer_params.get(constraint_param)

        # Lida com o caso especial de 'k_h * k_w' para VVAU
        if constraint_param == "k_h * k_w":
            k_h = layer_params.get("k_h")
            k_w = layer_params.get("k_w")
            if k_h is not None and k_w is not None:
                constraint_param_value = k_h * k_w
            else:
                constraint_param_value = None
        
        # Lida com a regra de parallel_window para ConvolutionInputGenerator
        if op_type.startswith("ConvolutionInputGenerator") and layer_params.get("parallel_window", 0) == 1:
            self.logger.debug("Regra de divisibilidade ignorada para parallel_window = 1")
            return current_value

        if constraint_param_value is None:
            self.logger.warning(f"Parâmetro de restrição '{constraint_param}' não encontrado para a camada '{layer_name}'. Retornando o teto do valor ideal.")
            return current_value + 1

        # Encontra o próximo valor válido
        next_valid = current_value + 1
        
        if constraint_param_value > 0:
            self.logger.debug(f"teste {constraint_param_value}")
            self.logger.debug(f"teste {current_value}")
            while True:
                if constraint_param_value % next_valid == 0:
                    self.logger.debug(f"return {next_valid}")
                    return next_valid
                next_valid += 1
                if next_valid > constraint_param_value:
                    self.logger.debug(f"return {constraint_param_value}")
                    return constraint_param_value # Retorna o valor máximo para indicar que o limite foi atingido
            
    def plot_bottleneck_evolution(self, iterations: list, data_cycles: list, bottleneck_names: list, all_formulas: dict):
        """Plota a evolução do gargalo a cada iteração, com cores e traços dinâmicos."""
        self.logger.info("Gerando gráfico de evolução do gargalo...")
        
        f_clock = 100e6 # 100 MHz
        data_fps = [f_clock / c if c > 0 else 0 for c in data_cycles]

        fig, ax = plt.subplots(figsize=(12, 9)) # Aumenta a altura da figura para a legenda

        # Mapeia cada função para uma cor e um traço únicos
        unique_bottlenecks = list(all_formulas.keys())
        colors = plt.cm.get_cmap('viridis', len(unique_bottlenecks))
        linestyles = ['-', '--', '-.', ':']
        
        color_map = {name: colors(i) for i, name in enumerate(unique_bottlenecks)}
        linestyle_map = {name: linestyles[i % len(linestyles)] for i, name in enumerate(unique_bottlenecks)}
        
        # Legenda customizada com os coeficientes
        legend_elements = []
        for name, data in all_formulas.items():
            # Tenta extrair K. Se a fórmula não for simples, usa 1.0 como padrão
            try:
                if "MVAU" in data['op_type']:
                    formula_k = data.get("MH", 1) * data.get("MW", 1) * data.get("numInputVectors", 1)
                else:
                    formula_k = float(data['formula'].split('/')[0].replace("(", "").replace(")", ""))
            except (ValueError, IndexError):
                formula_k = 1.0
            label = f"{name}: K = {formula_k:,.0f}"
            legend_elements.append(Line2D([0], [0], color=color_map[name], lw=4, linestyle=linestyle_map[name], label=label))
            
        # MODIFICAÇÃO: Posiciona a legenda abaixo do gráfico
        ax.legend(
            handles=legend_elements,
            loc='upper center',          # Ancoragem no topo e centro da legenda
            bbox_to_anchor=(0.5, -0.15), # Posição: 50% horizontal (centro), -15% vertical (abaixo)
            ncol=3,                      # Número de colunas para otimizar o espaço
            fancybox=True,
            shadow=True,
            title="Camadas e Coeficientes"
        )

        # Plota os segmentos de linha com cores e traços dinâmicos
        for i in range(len(iterations) - 1):
            current_bottleneck = bottleneck_names[i]
            color = color_map.get(current_bottleneck, 'gray')
            linestyle = linestyle_map.get(current_bottleneck, '-')
            ax.plot(
                iterations[i:i+2],
                data_fps[i:i+2],
                color=color,
                linewidth=3,
                marker='o',
                linestyle=linestyle
            )
            
        # Adiciona a última linha para o ponto final
        if len(iterations) > 1:
            ax.plot(
                iterations[-2:],
                data_fps[-2:],
                color=color_map.get(bottleneck_names[-1], 'gray'),
                linewidth=3,
                marker='o',
                linestyle=linestyle_map.get(bottleneck_names[-1], '-')
            )

        ax.set(
            xlabel="Iteração",
            ylabel="FPS (Frames por Segundo)",
            title="Evolução do Gargalo por Iteração (Otimização Incremental)",
            ylim=(0, None)
        )
        ax.grid(True, linestyle=':')
        ax.ticklabel_format(style='plain', axis='y')
        
        # MODIFICAÇÃO: Ajusta o layout para dar espaço para a legenda
        fig.subplots_adjust(bottom=0.25) # Aumenta a margem inferior para caber a legenda
        
        plt.savefig("bottleneck_evolution.png")


def modify_folding_analytical(current_folding, analyzer, cycle_formulas):
    import copy
    new_folding = copy.deepcopy(current_folding)
    
    layer_cycles = {}
    for layer_name, data in cycle_formulas.items():
        formula = data['formula']
        cfg = current_folding.get(layer_name, {})
        defaults = current_folding.get("Defaults", {"PE": 1, "SIMD": 1})
        
        pe = cfg.get("PE", defaults.get("PE", 1))
        simd = cfg.get("SIMD", defaults.get("SIMD", 1))
        current_params = {"PE": pe, "SIMD": simd}
        
        if "ConvolutionInputGenerator" in data.get("op_type", "") and cfg.get("parallel_window", 0) == 1:
            cycles = 2
        else:
            cycles = analyzer._eval_formula(formula, current_params)
        
        layer_cycles[layer_name] = cycles

    if not layer_cycles:
        return new_folding

    sorted_layers = sorted(layer_cycles.items(), key=lambda item: item[1], reverse=True)
    bottleneck_name, bottleneck_cycles = sorted_layers[0]

    if bottleneck_cycles <= 1:
        return new_folding

    data = cycle_formulas.get(bottleneck_name, {})
    op_type = data.get("op_type", "")
    cfg = new_folding.get(bottleneck_name)
    if not cfg:
        new_folding[bottleneck_name] = {"PE": 1, "SIMD": 1, "ram_style": "auto", "resType": "auto"}
        cfg = new_folding[bottleneck_name]

    current_pe = cfg.get("PE", 1)
    current_simd = cfg.get("SIMD", 1)

    if "SIMD" in data.get("formula", ""):
        if "MVAU" in op_type or "MatrixVectorActivation" in op_type:
            next_simd = analyzer._find_next_valid_parallelism(bottleneck_name, current_simd, op_type, data, "SIMD")
            if next_simd > current_simd:
                cfg["SIMD"] = next_simd
            else:
                next_pe = analyzer._find_next_valid_parallelism(bottleneck_name, current_pe, op_type, data, "PE")
                if next_pe > current_pe:
                    cfg["PE"] = next_pe
        elif "ConvolutionInputGenerator" in op_type:
            next_simd = analyzer._find_next_valid_parallelism(bottleneck_name, current_simd, op_type, data, "SIMD")
            if next_simd > current_simd:
                cfg["SIMD"] = next_simd
            else:
                if cfg.get("parallel_window", 0) == 0:
                    cfg["parallel_window"] = 1
        else:
            next_simd = analyzer._find_next_valid_parallelism(bottleneck_name, current_simd, op_type, data, "SIMD")
            if next_simd > current_simd:
                cfg["SIMD"] = next_simd
    elif "PE" in data.get("formula", ""):
        next_pe = analyzer._find_next_valid_parallelism(bottleneck_name, current_pe, op_type, data, "PE")
        if next_pe > current_pe:
            cfg["PE"] = next_pe

    return new_folding


def _get_base_onnx(request_path, base_build_dir):
    import json, yaml, os
    from utils.hw_utils import get_finn_ready_model
    import run_fps_map_job
    
    with open(request_path, 'r') as f:
        request_data = json.load(f)
        
    model_id = request_data.get('model_id')
    fpga_part = request_data.get('fpga_part', 'xc7z020clg400-1')
    
    with open('models/registry_models.yaml', 'r') as f:
        model_registry = yaml.safe_load(f)
        
    model_info = model_registry.get(model_id)
    quant = model_info.get("quant", model_info.get("weight_quant"))
    
    logger.info("Preparando ONNX principal (build zero)...")
    master_onnx_path = get_finn_ready_model(model_info, base_build_dir)
    
    est_dir_1 = run_fps_map_job._run_estimate_build(
        base_build_dir, master_onnx_path, "run0_get_initial_fold", 
        model_info.get("topology_id"), quant, fpga_part, target_fps=1
    )
    
    intermediate_onnx_path = os.path.join(est_dir_1, "intermediate_models", "step_generate_estimate_reports.onnx")
    auto_folding_json_path = os.path.join(est_dir_1, "auto_folding_config.json")
    return intermediate_onnx_path, auto_folding_json_path


def analyze_and_plot(onnx_model_path, debug_mode):
    """Executa a lógica original de simular 20-100 iterações em structs efêmeros para plotar FPS."""
    try:
        analyzer = FinnCycleEstimator(onnx_model_path, debug=debug_mode)
        cycle_formulas = analyzer.get_cycle_formulas()
        if not cycle_formulas:
            logger.info("Nenhuma camada de processamento foi encontrada. Análise encerrada.")
            return

        simd_state = {name: 1 for name, data in cycle_formulas.items() if "SIMD" in data.get("formula", "")}
        pe_state = {name: 1 for name, data in cycle_formulas.items() if "PE" in data.get("formula", "")}
        
        for name in cycle_formulas:
            if "ConvolutionInputGenerator" in cycle_formulas[name].get("op_type", ""):
                cycle_formulas[name]["parallel_window"] = 0
                
        bottleneck_data_iterations, bottleneck_data_cycles, bottleneck_data_names = [], [], []
        last_bottleneck_name, last_bottleneck_cycles = None, np.inf
        layer_cycles = {name: 0.0 for name in cycle_formulas.keys()}

        iteration = 1
        while True:
            for layer_name, data in cycle_formulas.items():
                formula = data['formula']
                current_params = {"PE": pe_state.get(layer_name, 1), "SIMD": simd_state.get(layer_name, 1)}
                
                if "ConvolutionInputGenerator" in data.get("op_type", "") and data.get("parallel_window", 0) == 1:
                    cycles = 2
                else:
                    cycles = analyzer._eval_formula(formula, current_params)
                layer_cycles[layer_name] = cycles
            
            sorted_layers = sorted(layer_cycles.items(), key=lambda item: item[1], reverse=True)
            bottleneck_name, bottleneck_cycles = sorted_layers[0]

            bottleneck_data_iterations.append(iteration)
            bottleneck_data_cycles.append(bottleneck_cycles)
            bottleneck_data_names.append(bottleneck_name)
            
            if bottleneck_cycles <= 1 or (bottleneck_name == last_bottleneck_name and np.isclose(bottleneck_cycles, last_bottleneck_cycles)):
                logger.info(f"\n[SUCESSO] Otimização analítica concluída (Limite atingido).")
                break
            
            last_bottleneck_name = bottleneck_name
            last_bottleneck_cycles = bottleneck_cycles

            logger.info(f"\n--- Iteração {iteration} --- Gargalo: {bottleneck_name} com {bottleneck_cycles:.2f} ciclos.")
            data = cycle_formulas.get(bottleneck_name, {})
            op_type = data.get('op_type')
            is_parallel_window_active = data.get("parallel_window", 0) == 1

            if "SIMD" in data.get("formula", ""):
                current_pe, current_simd = pe_state.get(bottleneck_name, 1), simd_state.get(bottleneck_name, 1)

                if "MVAU" in op_type:
                    next_simd_to_try = analyzer._find_next_valid_parallelism(bottleneck_name, current_simd, op_type, data, "SIMD")
                    if next_simd_to_try > current_simd:
                        simd_state[bottleneck_name] = next_simd_to_try
                    else:
                        next_pe_to_try = analyzer._find_next_valid_parallelism(bottleneck_name, current_pe, op_type, data, "PE")
                        if next_pe_to_try > current_pe: pe_state[bottleneck_name] = next_pe_to_try
                        else: break
                elif "ConvolutionInputGenerator" in op_type:
                    next_simd_to_try = analyzer._find_next_valid_parallelism(bottleneck_name, current_simd, op_type, data, "SIMD")
                    if next_simd_to_try > current_simd: simd_state[bottleneck_name] = next_simd_to_try
                    else:
                        if not is_parallel_window_active: cycle_formulas[bottleneck_name]["parallel_window"] = 1
                        else: break
                else:
                    next_simd_to_try = analyzer._find_next_valid_parallelism(bottleneck_name, current_simd, op_type, data, "SIMD")
                    if next_simd_to_try > current_simd: simd_state[bottleneck_name] = next_simd_to_try
                    else: break
            
            if "ConvolutionInputGenerator" in data.get("op_type", "") and 'next_simd_to_try' in locals() and next_simd_to_try <= current_simd and not is_parallel_window_active:
                iteration += 1; continue

            f_clock = 100e6
            bottleneck_fps = f_clock / bottleneck_cycles
            logger.info(f"FPS do gargalo: {bottleneck_fps:.2f} FPS.")
            iteration += 1
            
        bottleneck_data_iterations.pop()
        bottleneck_data_cycles.pop()
        bottleneck_data_names.pop()
        analyzer.plot_bottleneck_evolution(bottleneck_data_iterations, bottleneck_data_cycles, bottleneck_data_names, cycle_formulas)

    except Exception:
        logger.critical("Ocorreu um erro inesperado:", exc_info=True)


def _evaluate_bottleneck_fps(folding_config, analyzer, cycle_formulas):
    max_cycles = 1
    for layer_name, data in cycle_formulas.items():
        formula = data['formula']
        cfg = folding_config.get(layer_name, {})
        current_params = {"PE": cfg.get("PE", 1), "SIMD": cfg.get("SIMD", 1)}
        
        if "ConvolutionInputGenerator" in data.get("op_type", "") and cfg.get("parallel_window", 0) == 1:
            cycles = 2
        else:
            cycles = analyzer._eval_formula(formula, current_params)
            
        if cycles > max_cycles:
            max_cycles = cycles
            
    f_clock = 100e6
    return f_clock / max_cycles if max_cycles > 0 else 0


def generate_foldings(onnx_model_path, auto_folding_path, output_dir, debug_mode):
    """Executa a lógica de exportação estruturada de .json por step do modelo analítico."""
    import json, os, csv
    
    analyzer = FinnCycleEstimator(onnx_model_path, debug=debug_mode)
    cycle_formulas = analyzer.get_cycle_formulas()

    if not cycle_formulas:
        logger.error("[✗] Nenhuma fórmula de Hardware pôde ser extraída do ONNX.")
        return

    logger.info("Gerando folding base inicial a partir do template do FINN...")
    
    if auto_folding_path and os.path.exists(auto_folding_path):
        with open(auto_folding_path, "r") as f:
            current_folding = json.load(f)
    else:
        from utils.hw_utils import utils
        current_folding = utils.reset_folding({}, onnx_model_path)
    
    foldings_dir = os.path.join(output_dir, "foldings")
    os.makedirs(foldings_dir, exist_ok=True)
    
    csv_rows = [["Step", "FPS", "Configuration_File"]]
    
    step = 1
    fps = _evaluate_bottleneck_fps(current_folding, analyzer, cycle_formulas)
    filename = f"folding_step_{step:03d}.json"
    csv_rows.append([step, f"{fps:.2f}", filename])
    
    out_path = os.path.join(foldings_dir, filename)
    with open(out_path, "w") as f:
        json.dump(current_folding, f, indent=2)
    logger.info(f"[✓] Step {step}: {out_path} salvo com {fps:.2f} FPS.")

    while True:
        step += 1
        new_folding = modify_folding_analytical(current_folding, analyzer, cycle_formulas)
        
        if new_folding == current_folding:
            logger.info(f"\n[✓] Otimização analítica chegou ao limite de paralelismo.")
            logger.info(f"Total de {step-1} configurations geradas na pasta {foldings_dir}")
            break
            
        current_folding = new_folding
        fps = _evaluate_bottleneck_fps(current_folding, analyzer, cycle_formulas)
        filename = f"folding_step_{step:03d}.json"
        csv_rows.append([step, f"{fps:.2f}", filename])
        
        out_path = os.path.join(foldings_dir, filename)
        with open(out_path, "w") as f:
            json.dump(current_folding, f, indent=2)
        logger.debug(f"[✓] Step {step}: {out_path} salvo com {fps:.2f} FPS.")

    csv_path = os.path.join(output_dir, "analytical_fps_map.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    logger.info(f"[✓] Mapa CSV de FPS (analytical_fps_map.csv) exportado em {output_dir}")


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Utilitário Analítico de Estimação de Ciclos FINN-HARA.")
    parser.add_argument('--onnx', type=str, help="Caminho para o ONNX do modelo (opcional se --request for usado).")
    parser.add_argument('--request', type=str, help="Caminho do request.json (gera o ONNX automaticamente subindo um build0).")
    parser.add_argument('--debug', action='store_true', help="Ativa logs iterativos detalhados de extração e cálculos.")
    parser.add_argument('--generate-folding-files', action='store_true', help="Se ativado, irá exportar arquivos .json de configuração ao invés de apenas plotar.")
    parser.add_argument('--output-dir', type=str, default="analytical_foldings_output", help="Pasta para salvar os .json gerados (apenas com generate flags).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG, format='[%(levelname)s] %(message)s')
    
    onnx_path = args.onnx
    auto_folding_path = None
    
    if args.request:
        if not os.path.exists(args.request):
            logger.error(f"Arquivo de request ausente: {args.request}")
            exit(1)
        out_b = args.output_dir if args.generate_folding_files else "tmp_analy_build"
        os.makedirs(out_b, exist_ok=True)
        onnx_path, auto_folding_path = _get_base_onnx(args.request, out_b)
        if not onnx_path or not os.path.exists(onnx_path):
            logger.error(f"Falha gravíssima ao obter onnx base a partir do FINN (run0).")
            exit(1)
            
    if not onnx_path:
        logger.error("Você deve especificar --onnx ou --request para fornecer o modelo Base.")
        exit(1)

    if args.generate_folding_files:
        generate_foldings(onnx_path, auto_folding_path, args.output_dir, args.debug)
    else:
        analyze_and_plot(onnx_path, args.debug)