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





#do wizualizacji uzywam grafow z pythona : https://miroslawmamczur.pl/018-diagram-sieci-network-graph/ oraz https://stackoverflow.com/questions/20133479/how-to-draw-directed-graphs-using-networkx-in-python i innych
#klasa ta wizualizauje wylacznie siec neuronow - nie ma tu zadnych obliczen,
#oraz obrazek bedzie zawsze taki sam jak z zadania



