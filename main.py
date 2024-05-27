import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



#kalsa Neuron z zajec :
# init do inicializacji
# call do obliczenia wartosci neuronu (dzieki temu dziala jak funckja i przyjmuje wartosci wejsciowe )
class Neuron:
    def __init__(self, n_inputs, bias = 0., weights = None):
        self.b = bias
        if weights: self.ws = np.array(weights)
        else: self.ws = np.random.rand(n_inputs)

    def _f(self, x): #funkcja aktywacji - zwraca wartosc x jesli x > 0, w przeciwnym wypadku zwraca 0.1*x
        return max(x*.1, x)

    def __call__(self, xs): #z filmiku mnozenie wektora wejsciowego przez wektor wag, dodanie biasu i zastosowanie funkcji aktywacji wszytko w macierzach
        return self._f(xs @ self.ws + self.b)



#lsita perceptronow to ebdzize 2d, bo dodamy warswy
perceptrons = []

#ilsoc neuronow w warsiwe
neurons_in_layers = [3, 4, 4, 1]

#dodanie perceptronow do listy wagi same sie tworza w  kalsie neuron
for num_neurons in neurons_in_layers:
    layer = []
    for _ in range(num_neurons):
        if len(perceptrons) == 0:
            #peirwszwe nie maja wejsc
            perceptron = Neuron(n_inputs=0)
        else:
            # Tworzenie perceptronów z wejściami będącymi wyjściami z perceptronów poprzedniej warstwy
            inputs_to_perceptron = len(perceptrons[-1])
            perceptron = Neuron(n_inputs=inputs_to_perceptron)
        layer.append(perceptron)
    perceptrons.append(layer)

#test
# output = perceptrons[1][1]([0.5, 1.5, 2.5])
# print(output)



#do wizualizacji uzywam grafow z pythona : https://miroslawmamczur.pl/018-diagram-sieci-network-graph/ oraz https://stackoverflow.com/questions/20133479/how-to-draw-directed-graphs-using-networkx-in-python i innych
#klasa ta wizualizauje wylacznie siec neuronow - nie ma tu zadnych obliczen,
#oraz obrazek bedzie zawsze taki sam jak z zadania

def Net_visualisation(perceptrons):
    G = nx.DiGraph()

    for i, layer in enumerate(perceptrons):
        for j, perceptron in enumerate(layer):
            name = f"Layer {i} Neuron {j}"
            G.add_node(name, layer=i)

    # Dodanie krawędzi między neuronami w kolejnych warstwach
    for i in range(len(perceptrons) - 1):
        current_layer = perceptrons[i]
        next_layer = perceptrons[i + 1]
        for j, current_perceptron in enumerate(current_layer):
            for k, next_perceptron in enumerate(next_layer):
                current_node_name = f"Layer {i} Neuron {j}"
                next_node_name = f"Layer {i + 1} Neuron {k}"
                G.add_edge(current_node_name, next_node_name)

    # Rysowanie grafu
    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="green", font_size=8, font_weight="bold",
            arrowsize=20)
    plt.title("Visualizing Neural Network")
    plt.show()

Net_visualisation(perceptrons)




