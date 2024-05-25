import numpy as np
import pandas as pd
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



#neuron z 3 wejsciami
perceptron = Neuron(n_inputs=3, bias=-0.1, weights=[0.7, 0.6, 1.4])

# wektor wejsciowy i obliczenie wyjscia
output = perceptron([0.5, 0.3, 0.2])

# Print the details of the perceptron and the output
print("Perceptron:")
print(f"  Weights: {perceptron.ws}")
print(f"  Bias: {perceptron.b}")
print(f"  Input vector: {[0.5, 0.3, 0.2]}")
print(f"  Output: {output}")


#do wizualizacji uzywam grafow z pythona : https://miroslawmamczur.pl/018-diagram-sieci-network-graph/ oraz https://stackoverflow.com/questions/20133479/how-to-draw-directed-graphs-using-networkx-in-python i innych
#klasa ta wizualizauje wylacznie siec neuronow - nie ma tu zadnych obliczen,
#oraz obrazek bedzie zawsze taki sam jak z zadania

def Network_visualize():
    G = nx.DiGraph()
    G.add_nodes_from(["Input_1", "Input_2", "Input_3"], layer=0)
    G.add_nodes_from(["Hidden1_1", "Hidden1_2", "Hidden1_3", "Hidden1_4"], layer=1)
    G.add_nodes_from(["Hidden2_1", "Hidden2_2", "Hidden2_3", "Hidden2_4"], layer=2)
    G.add_nodes_from(["Output"], layer=3)
    nx.draw(G)
    plt.show()

Network_visualize()



