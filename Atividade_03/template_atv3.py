import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''
Informações sobre o como fazer a atividade:

* O script deverá ter o formato: seunome_sobrenome.py (sem espaços)
  Para pessoas com multiplos sobrenomes, colocar apenas o ultimo.
  Caracteres com acentos devem ser substituidos pelo equivalente sem acento.
  
* Não adicione nada no script fora dos locais onde está escrito :
    ### Seu código inicia aqui ###
    print("Olá mundo!")
    ### Seu código termina aqui ###
    
Esta atividade aborda o uso de redes neurais com TensorFlow e Keras utilizando
o dataset Fashion MNIST. As referências principais são os notebooks
"01_V2_mnist_CNN.ipynb" e "01_mnist_CNN.ipynb".
'''

### Seu código inicia aqui ###

nome = ''  # coloque aqui o nome completo sem espaços (use '_' entre as palavras)
email = ''  # coloque aqui o seu email

### Seu código termina aqui ###


def load_fashion_mnist(normalize=True, limit=None):
    '''
    Carrega o dataset Fashion MNIST utilizando tf.keras.datasets.fashion_mnist.

    Parâmetros
    ----------
    normalize : bool
        Quando True, converta os dados para float32 e divida todos os pixels por 255.
    limit : int | None
        Quando definido, limite o número de amostras retornadas para treino e teste
        utilizando os primeiros `limit` elementos de cada conjunto.

    Retorno
    -------
    (x_train, y_train), (x_test, y_test) : tuple
        Tupla contendo dois pares de arrays numpy.
    '''
    x_train, y_train, x_test, y_test = None, None, None, None
    ### Seu código inicia aqui ###

    ### Seu código termina aqui ###
    return (x_train, y_train), (x_test, y_test)


def reshape_for_cnn(images):
    '''
    Recebe um array de imagens 3D (N, 28, 28) e devolve o array com um canal extra,
    resultando em uma forma (N, 28, 28, 1), adequado para camadas Conv2D.
    '''
    imagens_ajustadas = None
    ### Seu código inicia aqui ###

    ### Seu código termina aqui ###
    return imagens_ajustadas


def build_dense_classifier(input_shape, hidden_units, num_classes, dropout_rate=0.0):
    '''
    Cria um modelo Sequencial com camadas densas para classificação de imagens 28x28.

    Parâmetros
    ----------
    input_shape : tuple
        Formato da entrada no padrão (28, 28, 1) após o reshape para CNN.
    hidden_units : list[int]
        Lista com o número de neurônios de cada camada densa intermediária.
    num_classes : int
        Número de classes de saída (para Fashion MNIST, utilize 10).
    dropout_rate : float
        Taxa de dropout aplicada após cada camada densa intermediária.
    '''
    model = None
    ### Seu código inicia aqui ###

    ### Seu código termina aqui ###
    return model


def compile_model(model, learning_rate):
    '''
    Compila o modelo recebido utilizando o otimizador Adam com a learning rate
    informada, a função de perda sparse_categorical_crossentropy e a métrica accuracy.
    '''
    ### Seu código inicia aqui ###

    ### Seu código termina aqui ###
    return model


def train_for_one_epoch(model, x_train, y_train, batch_size, shuffle=True):
    '''
    Executa uma época de treinamento chamando model.fit com epochs=1.

    O retorno deve ser o objeto History produzido pelo método fit para que possamos
    inspecionar as métricas registradas durante o treino.
    '''
    history = None
    ### Seu código inicia aqui ###

    ### Seu código termina aqui ###
    return history


#################################################################################
#### Fim da Atividade 3 - O código abaixo é apenas para mostrar o resultado #####
#### Não altere nada abaixo desta linha                                     #####
#################################################################################

if __name__ == '__main__':
    print("=" * 60)
    print("TESTANDO SUAS FUNÇÕES - RELATÓRIO DE PROGRESSO - ATIVIDADE 3")
    print("=" * 60)

    total_testes = 0
    testes_passaram = 0
    x_train = None
    y_train = None
    x_test = None
    modelo = None

    # Teste 1: load_fashion_mnist - formato e normalização
    print("\n1. Testando função load_fashion_mnist...")
    try:
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
        if x_train is None or x_test is None:
            raise AssertionError("Função retornou None - implemente a função!")
        assert x_train.shape == (60000, 28, 28), f"Formato esperado: (60000, 28, 28), obtido: {x_train.shape}"
        assert x_test.shape == (10000, 28, 28), f"Formato esperado: (10000, 28, 28), obtido: {x_test.shape}"
        assert x_train.dtype == np.float32, f"Tipo esperado: float32, obtido: {x_train.dtype}"
        assert np.max(x_train) <= 1.0 and np.min(x_train) >= 0.0, "Os valores devem estar normalizados entre 0 e 1"
        assert y_train.shape == (60000,), f"Formato esperado: (60000,), obtido: {y_train.shape}"
        testes_passaram += 1
        print("   [OK] PASSOU - Dados carregados e normalizados corretamente!")
    except AssertionError as e:
        print(f"   [X] FALHOU - {e}")
    except Exception as e:
        print(f"   [!] ERRO - {e}")
    total_testes += 1

    # Teste 2: reshape_for_cnn - formato com canal
    print("\n2. Testando função reshape_for_cnn...")
    try:
        if x_train is None:
            raise AssertionError("Dependência não atendida: implemente load_fashion_mnist antes deste teste")
        amostras = x_train[:32]
        ajustado = reshape_for_cnn(amostras)
        if ajustado is None:
            raise AssertionError("Função retornou None - implemente a função!")
        assert ajustado.shape == (32, 28, 28, 1), f"Formato esperado: (32, 28, 28, 1), obtido: {ajustado.shape}"
        testes_passaram += 1
        print("   [OK] PASSOU - Tensores ajustados corretamente para CNN!")
    except AssertionError as e:
        print(f"   [X] FALHOU - {e}")
    except Exception as e:
        print(f"   [!] ERRO - {e}")
    total_testes += 1

    # Teste 3: build_dense_classifier - estrutura do modelo
    print("\n3. Testando função build_dense_classifier...")
    try:
        modelo = build_dense_classifier((28, 28, 1), [256, 128], 10, dropout_rate=0.2)
        if modelo is None:
            raise AssertionError("Função retornou None - implemente a função!")
        assert isinstance(modelo, keras.Model), "O retorno deve ser uma instância de keras.Model"
        assert len(modelo.layers) >= 3, "O modelo deve possuir ao menos Flatten, camadas densas e a saída"
        assert modelo.layers[-1].units == 10, "A camada de saída deve possuir 10 neurônios"
        assert modelo.layers[-1].activation.__name__ == 'softmax', "A ativação da saída deve ser softmax"
        testes_passaram += 1
        print("   [OK] PASSOU - Modelo denso criado corretamente!")
    except AssertionError as e:
        print(f"   [X] FALHOU - {e}")
    except Exception as e:
        print(f"   [!] ERRO - {e}")
    total_testes += 1

    # Teste 4: compile_model - compilação com métricas
    print("\n4. Testando função compile_model...")
    try:
        if modelo is None:
            raise AssertionError("Dependência não atendida: implemente build_dense_classifier antes deste teste")
        compile_model(modelo, learning_rate=1e-3)
        if not hasattr(modelo, 'optimizer') or modelo.optimizer is None:
            raise AssertionError("O modelo não parece ter sido compilado - verifique o método compile")
        assert modelo.loss == 'sparse_categorical_crossentropy', "A função de perda deve ser sparse_categorical_crossentropy"
        if x_train is None or y_train is None:
            raise AssertionError("Dependência não atendida: implemente load_fashion_mnist antes deste teste")
        subset_x = reshape_for_cnn(x_train[:64])
        subset_y = y_train[:64]
        history = modelo.fit(subset_x, subset_y, epochs=1, batch_size=32, verbose=0)
        history_keys = set(history.history.keys())
        assert history_keys & {'accuracy', 'sparse_categorical_accuracy'}, "A métrica accuracy deve estar presente"
        testes_passaram += 1
        print("   [OK] PASSOU - Modelo compilado corretamente!")
    except AssertionError as e:
        print(f"   [X] FALHOU - {e}")
    except Exception as e:
        print(f"   [!] ERRO - {e}")
    total_testes += 1

    # Teste 5: train_for_one_epoch - chamada do fit
    print("\n5. Testando função train_for_one_epoch...")
    try:
        if x_train is None or y_train is None:
            raise AssertionError("Dependência não atendida: implemente load_fashion_mnist antes deste teste")
        tf.random.set_seed(7)
        subset_x = reshape_for_cnn(x_train[:256])
        subset_y = y_train[:256]
        history = train_for_one_epoch(modelo, subset_x, subset_y, batch_size=64)
        if history is None:
            raise AssertionError("Função retornou None - implemente a função!")
        assert hasattr(history, 'history'), "O retorno deve ser o objeto History do Keras"
        assert len(history.history.get('loss', [])) == 1, "A função deve rodar exatamente uma época"
        testes_passaram += 1
        print("   [OK] PASSOU - Treinamento de uma época executado com sucesso!")
    except AssertionError as e:
        print(f"   [X] FALHOU - {e}")
    except Exception as e:
        print(f"   [!] ERRO - {e}")
    total_testes += 1

    print("\n" + "=" * 60)
    print("RELATÓRIO FINAL")
    print("=" * 60)
    porcentagem = (testes_passaram / total_testes) * 100
    print(f"Testes realizados: {total_testes}")
    print(f"Testes que passaram: {testes_passaram}")
    print(f"Taxa de sucesso: {porcentagem:.1f}%")

    if porcentagem == 100:
        print("*** Excelente! O pipeline básico de redes neurais está funcional.")
    elif porcentagem >= 80:
        print("*** Muito bom! Ajuste os detalhes restantes para concluir a atividade.")
    elif porcentagem >= 60:
        print("*** Você está no caminho certo! Revise as funções que ainda falharam.")
    elif porcentagem >= 40:
        print("*** Continue praticando! Revise o uso da API do Keras nos notebooks de apoio.")
    else:
        print("*** Releia os notebooks indicados e tente novamente. Você consegue!")

    print("\nLembre-se de utilizar os notebooks da aula como referência:")
    print("- Aula_08/01_V2_mnist_CNN.ipynb")
    print("- Aula_01/01_mnist_pos_aula.ipynb (se disponível na sua cópia da aula)")
    print("=" * 60)
