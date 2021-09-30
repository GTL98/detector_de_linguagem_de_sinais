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

### Etapa 1: Importar as Dependências
Para essa etapa, serão usadas as seguintes bibliotecas:

    - Tensorflow;
    
    - OpenCV;
    
    - Mediapipe;
    
    - Scikit-Learn;
    
    - Matplotlib e;
    
    - Keras.
    
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

Agora, usaremos o Matplotlib para vizualisar o último frame captado pelo loop. Com isso, poderemos ver como as landmarks se comportam. Como o OpenCV grava o frame em BGR, é necessário converter para RGB para deixar a visualização melhor. Mas antes disso, é necessário chamar a função **desenhar_landmarks()** passando o *frame* e *resultados* como parâmetros. As imgens geradas pelo Matplotlib ficam confusas com as marcações, mas podemos ter uma noção de como funciona o Mediapipe. Feito isso, vamos colocar tudo isso no loop **while**. Em **cv2.imshow()**, é preciso trocar o parâmetro *frame* por *imagem*. Se deixar com *frame*, será mostrado o vídeo na tela sem as landmarks, com o parâmetro *image* as landmarks aparecem em tempo real; uma vez que esse parâmetro foi tratado para ter as landmarks, o *frame* por sua vez não.

Com isso, terminamos a Etapa 2.

### Etapa 3: Extrair os Valores de Pontos-chaves

Começaremos essa etapa colocando os valores de *resultados* em um array NumPy. O que nos é retornado quando chamamos *resultados* é uma lista com os valores das posições das landmarks. Passar isso para um array NumPy facilitará muito para mexermos com esses dados. Para isso, usaremos um laço **for** para percorrer a lista de landmarks. Há três valores padrão a serem colocados no array: **X**, **Y**, **Z**. Feito isso, vamos adicionar esse array em uma lista. Mas, para facilitar a nossa vida, faremos tuso isso em uma *list comprehension*. Quando a lista é feita pela *list comprehension*, o Python gera uma lista para cada conjunto dos quatro valores, mas para a LSTM é ruim esse formato, portanto, deveremos passar ao final da *list comprehension* a função **flatten()** para que o Python junte tudo em um array só.

Quando a câmera não capta algum membro, é retornado um erro quando chamamos as landmarks do mesmo. Se for feita uma lista para pegar esses dados, o Python mostra um erro dizendo que é impossível pegar esses dados, uma vez que eles não existem. Como a rede neural não pode receber erros, teremos te criar um array com zeros quando isso acontecer. Para isso, basta usar condição **if-else**. Se existir valores de landmarks, adicionar ao array esses valores, caso não, adicionar zeros com o **np.zeros**. Como o *pose_landmarks* é o único que possui **visibility** como dado, não podemos deixar de colocá-lo. Mas como saber o tamanho do array para completar com zeros? Basta chamar o método *len* na landmark desejada e multiplicar pelo número de itens dentro dela. Todos serão multiplicados por 3, menos o *pose_landmarks* que possui 4 valores. E para facilitar a nossa vida, vamos colocar tudo isso dentreo de uma função. Ao final da função, devemos retornar um array com todos os arrays em um só, para isso usaremos o método **concatenate()** do NumPy.

A Etapa 3 está concluída.

### Etapa 4: Configurar as Pastas para a Coleção de Arrays

Nessa quarta etapa, vamos criar as pastas para guardar as nossas coleções de arrays feitas na Etapa 3. Começamos criando uma variável para armazenar o caminho da pasta que guardaremos os arrays. A segunda variável é um array NumPy com as ações que queremos que sejam detectadas, são somente três (olá, obrigado e amo você), mas podem ser mais ações. Passaremos para a IA capturar 30 vídeos contendo os dados. E por fim, fixaremos a quantidade de frames necessários para a IA saber o que estamos querendo dizer com os sinais, o valor é de 30 frames. A quantidade de dados coletadas é o número da quantidade de vídeos, mais a quantidade de frames, mais a quantidade de ações, mais a quantidade de dados no array. Isso dá um total de 4.487.400 dados!!!

O próximo passo é criar as pastas que vamos armazenar os dados. Para isso, vamos usar um loop **for** em nosso array de ações. Para cada ação dentro desse array, será criada uma pasta para guardar os dados de somente desta ação. Ao final desse loop, teremos as pastas presentes no array das ações, e dentro dessas pastas, subpastas com cada sequência da ação (o número de subpastas é determinado pela variável *num_videos*). Vale lembrar que deve utilizar um bloco **try** para evitar que o Python crie as pastas e subpastas se já existirem.

Com isso, concluímos a Etapa 4.

### Etapa 5: Coletar os Valores de Pontos-chaves para Treino e Teste

Começamos essa etapa copiando o código do loop **while** da Etapa 2, onde faremos algumas modificações. Ao invés de deixarmos a câmera ligada o tempo todo, ela tirará algumas fotos de tempos em tempos pelas ações descritas no array da etapa anterior. Ou seja, a câmera não ficará aberta o tempo todo, mas somente nos momentos para pegar as informações. É muito importante colocar uma pausa entre uma ação e outra para que seja possível se ajeitar para fazer as outras ações. O que estamos fazendo aqui é quando for o frame 0 da captura, um texto aparecerá na tela para informar o que será feito e teremos 2 segundos para nos ajeitar e fazer a ação.

O próximo passo é salvar o array em um arquivo, cada arquivo representando o array do determinado frame. Para isso vamos usar o método **save()** do NumPy. A extensão do arquivo é *.npy*. O que estará salvo nesses documentos é o array dos pontos-chaves de cada captura feita. Vale ressaltar que os parâmetros passados para o método **save()** são o caminho do arquivo e os dados, no caso os pontos-chaves.

A parte final desta etapa é a mais "chatinha" de ser feita, precisamos agora salvar os arrays. Para isso, se ajeite na cadeira e começe a fazer os sinais na frente de câmera. Se ficar muito demorado para fazer, diminua o tempo de espera em **cv2.waitKey()**. Com as capturas feitas, fechamos a Etapa 5.

### Etapa 6: Processar os Dados e Criar Rótulos e Recursos

Vamos importar a biblioteca de treino do Scikit-Learn para criar uma partição de teste e outra de treino. Além da Scikit-Learn, vamos importar a bilbioteca Keras. Ambas nos ajudarão com a parte de rótulos. O método **train_test_split()** do Scikit-Learn separará os dados em dados de treino e dados de teste, é possível fazer essa separação na mão, mas podemos acabar fazendo algo errado e prejudicar a IA. O método **to_categorical()** do Keras serve para codificar os dados para o *onehot-encoding*, isso é de suma importência para que a IA entenda os dados. Feito isso, vamos criar um dicionário com os nossos rótulos.

Com o dicionário criado, vamos pegar todos os dados gerados e estruturá-los. Para isso, vamos criar um mega array com todos os dados coletados dos pontos-chaves. No nosso caso, o mega array possui um *shape* de 90, 30, 1662, ou seja, 90 vídeos (pois são 30 vídeos por ação), 30 frames por video e 1662 pontos-chaves capturados. O número de vídeos aumenta ou diminui dependendo de quantos sinais você quer que a IA aprenda. Agora, vamos pré-processar os dados para que possamos trabalhar com eles.

Começamos armazenando o mega array na variável **X** e os rótulos na variável **y**. O método **to_cateorial()** será usado na variável **y**. É criada uma lista com valores 0 e 1, e dessa forma a IA sabe qual ação que conforme a sequência de 0 e 1. Isso é o *one-hot enconding*. Como mencionado anteriormente, usaremos o médoto **train_test_split()** para separar os dados de teste e treino. É necessário passar os arrays (**X** e **y**) e o tamanho dos dados de teste (**test_size**). Para o tamnho de teste, foi usado o valod e 0.05, isto é: 5% dos dados são destinádos ao conjunto de teste e os outros 95%, aos dados de treino.

E por incrível que pareça, é somente isso na Etapa 6.

### Etapa 7: Construir e Treinar a Rede Neural LSTM

Nessa etapa vamos construir e treinar a nossa IA e para isso vamos usar o Keras. Importaremos 4 módulos: **Sequential**, **LSTM**, **Dense** e **TensorBoard**. O primeiro módulo serve para criar o sistema de rede neural sequencial. O LSTM serve para contruir a nossa rede neural e detectar as ações que fazemos e junto com o LSTM, o módulo Dense é mais uma camada a ser colocada na rede neural. Por fim, o TensorBoard serve para fazer o registro para monitorar e rastrear o nosso modelo ao longo do treinamento. O passo seguinte é criar um diretório para armazenar esses registros.

Com tudo feito, vamos finalmente criar a nossa rede neural!

Usamos o módulo **Sequential** para fazer uma sequência de neurônios em nossa IA. Colocamos 3 **LSTM** dentro dela. Um ponto importante é passar o valor *True* para o parâmetro *return_sequences*. Sempre que estiver usando o LSTM com o TensorFlow e a camada seguinte precisar das informações da camada anterior, deve-se passar *True* para esse parâmetro. Na última camada de LSTM deve-se informar *False* para esse parâmetro. E para o parâmetro *input_shape* devemos passar uma tupla com o número de vídeos (30) e o número de pontos-chaves (1662) e esse último parâmetro só vale para o primeiro LSTM.

Para a **Dense**, mudaremos a última camada com o parâmetro *actions.shape[0]*, que o valor é 3. Isso significa que teremos somente 3 saídas, que são os 3 sinais que usamos (olá, obrigado e amo você). Usando o *activation* como *softmax*, será retornado para nós a probabilidade do sinal que a câmera está capturando. Por exemplo, se for retornado esses valores: 0.7, 0.2, 0.1 a rede neural verá qual o maior valor (0.7) e em qual posição ele está (0). Com essa última informação, a IA vai até a lista das ações e vê qual é o item na posição recebida, e dessa forma será mostrado qual é a ação (olá).

Vamos agora compilar o modelo.
