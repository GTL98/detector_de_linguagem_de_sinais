# Rede Neural para detectar linguagem de sinais (achar um nome depois)

***Usar o Jupyter Notebook para esse projeto***

 ### Objetivo: Detectar em tempo real a linguagem de sinais.
  
    1 - Usar pontos-chaves MP (mediapipe) holísticos para a rede neural ser
    capaz de reconhecer as mãos e o rosto;
    
    2 - Treinar um modelo LSTM (Mémoria de curto prazo longa) com Tensorflow e Keras e;
  
    3 - Realizar predições em tempo real.
  

### Como vai funcionar:

    1 - Coletar os pontos-chaves MP holísticos;
    
    2 - Treinar uma rede neural com LSTM com camadas e;
    
    3 - Retornar em tempo real a detecção dos sinais com OpenCV.


O **detector** será feito em 11 etapas.

### Etapa 1: Importar e Instalar as Dependências
Para essa etapa, serão usadas as seguintes bibliotecas:

    - Tensorflow;
    
    - OpenCV;
    
    - Mediapipe;
    
    - Scikit-Learn e;
    
    - Matplotlib.
    
A única biblioteca a ser instalada foi a Tensorflow via **pip** pelo prompt de comando do Anaconda.
O Tensorflow será usado para rede neural, o OpenCv para trabalhar com a câmera do computador, o Mediapipe para extrair os
pontos-chaves holísticos, o Scikit-Learn para as métricas de avaliação, bem como para ser usado nos treinos e testes e o
Matplotlib para vizualisar as imagens de uma maneira mais fácil.

As bibliotecas importadas foram:

    - cv2 (OpenCV);
    - numpy;
    - os;
    - matplotlib;
    - time e;
    - mediapipe.
