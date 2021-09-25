# Rede Neural para detectar linguagem de sinais (achar um nome depois)

***Usar o Jupyter Notebook para esse projeto.***

***Todo o código estará no repositório.***

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

### Etapa 2: Pontos-chaves usando o MP Holístico

Nessa etapa, vamos acessar a câmera do computador com o OpenCV e detectar o que está na frente dela com o Mediapipe.

O primeiro passo aqui é usar o OpenCV para acessar a câmera ver como ela está se comportando. Informando o valor 0 para
cv2.VideoCapture(), está dizendo ao OpenCV para usar a câmera do computador. Depois disso, abriremos um loop **while**
para que a câmera continue aberta, leia o que está captando e mostre na tela em tempo real. Quando estamos usando o
método **read()** do OpenCV, ele retorna 2 valores que podemos armazenar. Um deles é armazenado na variável *frame*.
Como um vídeo nada mais é do que vários frames seguidos, essa variável de mesmo nome armazena o frame capturado e
mostra na tela e como este pedaço do código está em um loop, vários frames são exibidos em sequência, formando um
vídeo ao final.
