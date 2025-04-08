import json

class HardwareConfig:
    def __init__(self, json_data):
        self.modules = {}  # Estrutura para armazenar os módulos
        self.load_json(json_data)

    def load_json(self, json_data):
        """Carrega o JSON em uma estrutura interna."""
        for key, value in json_data.items():
            self.modules[key] = value  # Mantém os dados estruturados

    def show_matrix_view(self):
        """Exibe os módulos e seus principais parâmetros como uma matriz formatada."""
        headers = ["Module", "PE", "SIMD", "ram_style", "resType", "mem_mode"]
        col_widths = [30, 4, 6, 12, 8, 12]
        
        header_row = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
        print(header_row)
        print("-" * sum(col_widths))
        
        for module, params in self.modules.items():
            pe = str(params.get("PE", "-"))
            simd = str(params.get("SIMD", "-"))
            ram_style = params.get("ram_style", "-" )
            res_type = params.get("resType", "-" )
            mem_mode = params.get("mem_mode", "-" )
            
            row = f"{module:<{col_widths[0]}} | {pe:<{col_widths[1]}} | {simd:<{col_widths[2]}} | {ram_style:<{col_widths[3]}} | {res_type:<{col_widths[4]}} | {mem_mode:<{col_widths[5]}}"
            print(row)

    def update_parameter(self, module_name, param, value):
        """Atualiza um parâmetro específico dentro de um módulo."""
        if module_name in self.modules:
            self.modules[module_name][param] = value
            print(f"Atualizado {module_name} -> {param}: {value}")
        else:
            print(f"Módulo {module_name} não encontrado.")

    def export_json(self, filename="updated_config.json"):
        """Salva as alterações de volta para um arquivo JSON."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.modules, f, indent=4)
        print(f"Configuração salva em {filename}")


# Exemplo de uso:
if __name__ == "__main__":
    # Carregar JSON de um arquivo
    with open("folding.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    config = HardwareConfig(data)
    config.show_matrix_view()  # Exibe a matriz de módulos
    
    # Modificar alguns valores
    config.update_parameter("MatrixVectorActivation_1", "PE", 4)
    config.update_parameter("ConvolutionInputGenerator_2", "SIMD", 8)
    
    # Exportar JSON atualizado
    config.export_json("updated_config.json")
