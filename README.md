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
    
A única biblioteca a ser instalada foi a Tensorflow via **pip** pelo prompt de comando do Anaconda. O Tensorflow será usado para rede neural, o OpenCv para trabalhar com a câmera do computador, o Mediapipe para extrair os pontos-chaves holísticos, o Scikit-Learn para as métricas de avaliação, bem como para ser usado nos treinos e testes e o Matplotlib para vizualisar as imagens de uma maneira mais fácil.

As bibliotecas importadas foram:

    - cv2 (OpenCV);
    - numpy;
    - os;
    - matplotlib;
    - time e;
    - mediapipe.

### Etapa 2: Pontos-chaves usando o MP Holístico

Nessa etapa, vamos acessar a câmera do computador com o OpenCV e detectar o que está na frente dela com o Mediapipe.

O primeiro passo aqui é usar o OpenCV para acessar a câmera ver como ela está se comportando. Informando o valor 0 para cv2.VideoCapture(), está dizendo ao OpenCV para usar a câmera do computador. Depois disso, abriremos um loop **while** para que a câmera continue aberta, leia o que está captando e mostre na tela em tempo real. Quando estamos usando o método **read()** do OpenCV, ele retorna 2 valores que podemos armazenar. Um deles é armazenado na variável *frame*. Como um vídeo nada mais é do que vários frames seguidos, essa variável de mesmo nome armazena o frame capturado e mostra na tela e como este pedaço do código está em um loop, vários frames são exibidos em sequência, formando um vídeo ao final. Para encerrar esse loop, basta apertar a tecla **S**.

O segundo passo é criar o código do Mediapipe holístico e criar duas variáveis: um para o Mediapipe holístico e  a outra para o Mediapipe desenho. A primeira variável serve para fazer as detecções e a segunda, para desenhar essas detecções. Para facilitar nossas vidas, será criada uma função que recebe dois parâmetros: *imagem* e *modelo*. A primeira etapa dentro dessa função é converter a imagem recebida de BGR (**B**lue, **G**reen, **R**ed) para RGB (**R**ed, **G**reen, **B**lue). Em seguida, definiremos para não gravar essas imagens (isso economiza memória), detectamos a imagem, tornaremos gravável novamente e finalmente converteremos de RGB para BGR. O OpenCV por padrão captura as imagens no formato BGR, mas o Mediapipe aceita somente RGB; por isso essa conversão. Esse vai e volta deve ser feito para que o modelo do Mediapipe trabalhe com a imagem no formato RGB e quando voltar à tela, seja no formato do OpenCV (em BGR). Ao fim, retornaremos a imagem e o resultado do modelo. Com tudo certo na função, podemos adicioná-la ao código que abre a câmera, dentro do laço **while**. Mas antes disso, devemos instânciar todo esse laço em uma declaração **with** para acessarmos o nosso modelo holístico. É passado 2 parâmetros para o modelo: *min_detection_confidence* e *min_tracking_confidence*. O primeiro parâmetro serve para a detecção inicial e o segundo, para que o MP holístico ratreie os pontos-chaves detectados com o primeiro parâmetro. Ambos com o valor de 0,5.

O próximo passo é criar a função que vai desenhar as *landmarks*. As landmarks são as linhas para identificação do rosto, braços, mãos e tronco. Vamos passar dois parâmetros para essa função: *imagem* e *resultados*. O primeiro parâmetro é basicamente a imagem que a câmera está captando. O parâmetro *resultados* são os valores de landmarks retornado pela função **deteccao_mediapipe()**, onde possui os dados para desenhar as landmarks. Para cada item da função, são passados 5 parâmetros: *image*, *landmarks_list*, *connections*, *landmark_drawing_spec*  e *connection_drawing_spec*. O parâmetro *image* é um array NumPy com as informações das landmarks da imagem, basicamente isso. O segundo parâmetro é uma lista com todos os valores (X, Y e Z) das landmarks presentes na imagem. O parâmetro *connections* é uma lista indicando onde que uma landmark começa e termina juntando-se em outra. Por exemplo, a landmark do ombro direito termina no cotovelo direito, que por sua vez termina no pulso direito e assim por diante. O *landmarks_drawing_spec* serve para formatar o tamnaho e cor do s landmarks. O último parâmetro é a mesma coisa que o anterior, mas agora para as conexões entre os landmarks.

Agora, usaremos o Matplotlib para vizualisar o último frame captado pelo loop. Com isso, poderemos ver como as landmarks se comportam. Como o OpenCV grava o frame em BGR, é necessário converter para RGB para deixar a visualização melhor. Mas antes disso, é necessário chamar a função **desenhar_landmarks()** passando o *frame* e *resultados* como parâmetros. As imgens geradas pelo Matplotlib ficam confusas com as marcações, mas podemos ter uma noção de como funciona o Mediapipe. Feito isso, vamos colocar tudo isso no loop **while**. Em **cv2.imshow()**, é preciso trocar o parâmetro *frame* por *imagem*. Se deixar com *frame*, será mostrado o vídeo na tela sem as landmarks, com o parâmetro *image* as landmarks aparecem em tempo real; uma vez que esse parâmetro foi tratado para ter as landarks, o *frame* por sua vez não.
