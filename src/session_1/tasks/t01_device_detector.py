from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()
dispositivos = graph.get_input_devices()

for indice, nombre in enumerate(dispositivos):
    print(f"Índice [{indice}] -> {nombre}")