# https://www.tensorflow.org/guide/core/mlp_core

# 아래 코드를 읽기 전에
# 영어로 작성되어 있는 주석은 위 웹사이트에서 그대로 가져온 거야.
# 최대한 자세하게 작성했지만, 이해가 안 되는 부분은 GPT가 나보다 잘 알아
# 하지만 내가 잘못 작성한 것 같은 부분이 있으면 알려줘

# 그냥 분석하면서 느낀 건데
# NN.py는 좀 더 기초부터 알려주는 예제 코드, 직접 구현한 쓸데없는 부분 많음
# CNN.py는 NN에 있는 내용 다 알지? 그거 사용하고 레이어 종류 바꾸면 CNN됨 이런 느낌
# 좀 더 불친절하고 짧음

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import tempfile
import os

# matplotlib 그래프 창 크기 통일
# matplotlib이 뭔지 알아? 지난번에 했지만 까먹었을까봐. 그래프를 그릴 수 있게 해 주는 라이브러리야.
matplotlib.rcParams['figure.figsize'] = [9, 6]

import tensorflow as tf
import tensorflow_datasets as tfds

# 랜덤이 들어간 연산 시드 고정, 여러 번 코드 돌려도 같은 결과 나오게
tf.random.set_seed(22)

# MNIST 데이터 불러와서 자르기, 학습, 테스트 용도 분류
# val_data: validation, test_data: testing
train_data, val_data, test_data = tfds.load("mnist", 
                                            split=['train[10000:]', 'train[0:10000]', 'test'],
                                            batch_size=128, as_supervised=True)

# x_viz, y_viz matplotlib 그래프용 데이터 너무 많으면 처리하기 힘드니까 1500개만
# matplotlib 그래프에는 이 데이터 사용
x_viz, y_viz = tfds.load("mnist", split=['train[:1500]'], batch_size=-1, as_supervised=True)[0]
x_viz = tf.squeeze(x_viz, axis=3)

# ------------------------------------------------------

# 그래프_1
# 데이터 형식과 (28 * 28 흑백 숫자) 짝지어진 숫자 (True Label) 보여줌
for i in range(9):
    plt.subplot(3,3,1+i)
    plt.axis('off')
    plt.imshow(x_viz[i], cmap='gray')
    plt.title(f"True Label: {y_viz[i]}")
plt.subplots_adjust(hspace=.5)
plt.show()

# ------------------------------------------------------

# 그래프_2
# 숫자 분포 보여줌
sns.countplot(x=y_viz.numpy());
plt.xlabel('Digits')
plt.title("MNIST Digit Distribution");
plt.show()

# ------------------------------------------------------

# CNN과의 핵심 차이점!!
# CNN은 위치정보를 포함하지만, 
# NN은 [(0, 0), (1, 0), ... , (0, 1), (1, 1), ... ] 그냥 가로로 넓게 펼침
def preprocess(x, y):
  # Reshaping the data -> CNN과 다름
  x = tf.reshape(x, shape=[-1, 784])
  # Rescaling the data -> 밝기 정보를 0-1로 처리, 학습 용이
  x = x/255
  return x, y

train_data, val_data = train_data.map(preprocess), val_data.map(preprocess)

# ------------------------------------------------------

# 그래프_3
# ReLU 그래프 시각화
x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)
plt.plot(x, tf.nn.relu(x));
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.title('ReLU activation function');

# ------------------------------------------------------

# 그래프_4
# Softmax 그래프 시각화
x = tf.linspace(-4, 4, 201)
x = tf.cast(x, tf.float32)
plt.plot(x, tf.nn.softmax(x, axis=0));
plt.xlabel('x')
plt.ylabel('Softmax(x)')
plt.title('Softmax activation function');

# ------------------------------------------------------

# 가중치를 작은 값으로 설정 -> 학습 안정화
# Dense Layer에 사용될 부분
# tf.keras.initializers.GlorotUniform로 불러올 수 있지만, 학습을 위한 예제 코드인 만큼 직접 구현한 듯
# 가져올 수 있지만 직접 구현한 부분이 아주 많아. 사실 대부분이야. 세부적인 부분을 직접 작성할 수 있고, 연산을 줄일 수 있을 듯.

# 가중치 초기화를 구현. 학습에서 가중치 초기값이 너무 크면 활성화 값이 너무 커지고, 너무 작으면 신호가 사라져.
# 무슨 소리냐면, 우리가 학습을 최적화하는 걸 시각화할 때 울퉁불퉁한 언덕에 공을 굴리는 것처럼 시각화하잖아.
# 초기값이 너무 크다는 건 이상한 겁나 높은 언덕에 공을 던지는 거야. 학습이 잘 되기까지 안에서 튕기면서 난리가 나.
# 이걸 방지하기 위해 좀 적당한 언덕 위에 내려놓아야 하는 거지.
# 너무 작다는 건, 기울기가 거의 없는 언덕 위에 내려놓아서 초기 가속까지 시간이 오래 걸린다는 거야. 아니면 멈춰버릴 수도 있고.
# 자세한 수학 식은... 공부할 필요가 있을까?
def xavier_init(shape):
  # in_dim: 입력 차원, out_dim: 출력 차원
  # 예. shape가 차원이니까요.
  # Computes the xavier initialization values for a weight matrix
  in_dim, out_dim = shape

  # 수학 식 구현한 부분.
  xavier_lim = tf.sqrt(6.)/tf.sqrt(tf.cast(in_dim + out_dim, tf.float32))
  # 랜덤한 수 추출, 하지만 시드 설정해서 같은 값 나오도록 해뒀네.
  weight_vals = tf.random.uniform(shape=(in_dim, out_dim), 
                                  minval=-xavier_lim, maxval=xavier_lim, seed=22)
  return weight_vals

# Dense Layer 구현
# keras.layer.Dense 가장 기본적인 레이어
# 모든 입력 노드와 모든 출력 노드가 연결되어 있음.
class DenseLayer(tf.Module):

  def __init__(self, out_dim, weight_init=xavier_init, activation=tf.identity):
    # Initialize the dimensions and activation functions
    self.out_dim = out_dim
    self.weight_init = weight_init
    self.activation = activation
    self.built = False

  def __call__(self, x):
    if not self.built:
      # Infer the input dimension based on first call
      # 입력 크기 자동 추론. 그러니까 x.shape[1]을(차원 크기를) in_dim으로 설정함.
      self.in_dim = x.shape[1]
      # 가중치 W, 편향 b 초기화
      # Initialize the weights and biases
      self.w = tf.Variable(self.weight_init(shape=(self.in_dim, self.out_dim)))
      self.b = tf.Variable(tf.zeros(shape=(self.out_dim,)))
      self.built = True
    # Compute the forward pass
    # z = xW + b -> 활성화 함수 적용
    z = tf.add(tf.matmul(x, self.w), self.b)
    return self.activation(z)

# MLP 모델 정의
# MLP 모델은 그냥 신경망임. Dense Layer들을 여러 개 쌓은 신경망
class MLP(tf.Module):

    def __init__(self, layers):
        # 레이어들을 리스트로 받음. 이 아래 mlp_model을 정의한 곳에서 레이어를 직접 지정해 줘.
        self.layers = layers

    @tf.function
    def __call__(self, x, preds=False): 
    # Execute the model's layers sequentially
        for layer in self.layers:
            x = layer(x)
        return x

# 은닉층 크기 지정(매직넘버 방지)
hidden_layer_1_size = 700
hidden_layer_2_size = 500
# 출력층 크기가 10인 이유는 0~9 숫자 분류에 사용되는 모델이기 때문.
output_size = 10

# 손실 함수 & 정확도
mlp_model = MLP([
    DenseLayer(out_dim=hidden_layer_1_size, activation=tf.nn.relu),
    DenseLayer(out_dim=hidden_layer_2_size, activation=tf.nn.relu),
    DenseLayer(out_dim=output_size)]) # Softmax는 나중에 적용.

# 손실함수, 인터넷에 치면 수학 식 나옴
# 이건 식이 쉬워서 충분히 이해하고 보고서에 넣어도 될 듯.
# H = -sigma(p(x) * log p(x)) 이게 아마 제일 쉬운 식이고, 바리에이션이 많긴 함 워낙 자주 쓰는 식이라
def cross_entropy_loss(y_pred, y):
  # Compute cross entropy loss with a sparse operation
  sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
  return tf.reduce_mean(sparse_ce)

def accuracy(y_pred, y):
  # Compute accuracy after extracting class predictions
  # class_preds: 예측값.
  # y_pred. 그러니까 raw 숫자 예측값에 softmax를 씌워서 최종 예측값을 내.
  class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1)
  # 정답과 비교해서 같은지 확인
  is_equal = tf.equal(y, class_preds)
  return tf.reduce_mean(tf.cast(is_equal, tf.float32))

# Adam 직접 구현
# tf.keras.optimizers.Adam로 가져올 수 있음.
# 이미 배워서 알고 있겠지만 경사하강법과 관련된 거고, 식이 겁나 복잡함
# 이 부분 주석이 좀 단촐한데, 내가 이해를 못해서 그런거고 어짜피 기본적인 컨셉만 알고 있으면 될 듯. 대학가서 배웁시다
class Adam:

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7):
      # Initialize optimizer parameters and variable slots
      self.beta_1 = beta_1
      self.beta_2 = beta_2
      self.learning_rate = learning_rate
      self.ep = ep
      self.t = 1.   # 학습 단계 저장
      self.v_dvar, self.s_dvar = [], []
      self.built = False

    def apply_gradients(self, grads, vars):
      # Initialize variables on the first call
      if not self.built:
        for var in vars:
          v = tf.Variable(tf.zeros(shape=var.shape))
          s = tf.Variable(tf.zeros(shape=var.shape))
          self.v_dvar.append(v)
          self.s_dvar.append(s)
        self.built = True
      # Update the model variables given their gradients
      for i, (d_var, var) in enumerate(zip(grads, vars)):
        # 1, 2차 모멘트 업데이트
        self.v_dvar[i].assign(self.beta_1*self.v_dvar[i] + (1-self.beta_1)*d_var)
        self.s_dvar[i].assign(self.beta_2*self.s_dvar[i] + (1-self.beta_2)*tf.square(d_var))
        # 편향 보정
        v_dvar_bc = self.v_dvar[i]/(1-(self.beta_1**self.t))
        s_dvar_bc = self.s_dvar[i]/(1-(self.beta_2**self.t))
        # 가중치 업데이트
        var.assign_sub(self.learning_rate*(v_dvar_bc/(tf.sqrt(s_dvar_bc) + self.ep)))
      self.t += 1.
      return
    
# 학습 루프
# 한 번의 학습 단계: 모델에 데이터 넣기 -> 손실 계산하기 -> 기울기 계산하기 -> 가중치 업데이트하기(adam)
def train_step(x_batch, y_batch, loss, acc, model, optimizer):
  # Update the model state given a batch of data
  with tf.GradientTape() as tape:   # 미분을 위해 기록 시작
    y_pred = model(x_batch)
    batch_loss = loss(y_pred, y_batch)
  batch_acc = acc(y_pred, y_batch)
  grads = tape.gradient(batch_loss, model.variables)  # 기울기 계산
  optimizer.apply_gradients(grads, model.variables)   # 가중치 업데이트
  return batch_loss, batch_acc

def val_step(x_batch, y_batch, loss, acc, model):
  # Evaluate the model on given a batch of validation data
  # 검증 단계에서는 가중치를 바꾸지 않고 평가만 함.
  y_pred = model(x_batch)
  batch_loss = loss(y_pred, y_batch)
  batch_acc = acc(y_pred, y_batch)
  return batch_loss, batch_acc

def train_model(mlp, train_data, val_data, loss, acc, optimizer, epochs):
  # Initialize data structures
  train_losses, train_accs = [], []
  val_losses, val_accs = [], []

  # epoch는 학습 횟수인 거 알지?
  # Format training loop and begin training
  for epoch in range(epochs):
    batch_losses_train, batch_accs_train = [], []
    batch_losses_val, batch_accs_val = [], []

    # 학습
    # Iterate over the training data
    for x_batch, y_batch in train_data:
      # Compute gradients and update the model's parameters
      batch_loss, batch_acc = train_step(x_batch, y_batch, loss, acc, mlp, optimizer)
      # Keep track of batch-level training performance
      batch_losses_train.append(batch_loss)
      batch_accs_train.append(batch_acc)

    # 검증
    # Iterate over the validation data
    for x_batch, y_batch in val_data:
      batch_loss, batch_acc = val_step(x_batch, y_batch, loss, acc, mlp)
      batch_losses_val.append(batch_loss)
      batch_accs_val.append(batch_acc)

    # 한 epoch에서의 평균값 기록, 출력
    # Keep track of epoch-level model performance
    train_loss, train_acc = tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_accs_train)
    val_loss, val_acc = tf.reduce_mean(batch_losses_val), tf.reduce_mean(batch_accs_val)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch: {epoch}")
    print(f"Training loss: {train_loss:.3f}, Training accuracy: {train_acc:.3f}")
    print(f"Validation loss: {val_loss:.3f}, Validation accuracy: {val_acc:.3f}")
  return train_losses, train_accs, val_losses, val_accs

train_losses, train_accs, val_losses, val_accs = train_model(mlp_model, train_data, val_data, 
                                                             loss=cross_entropy_loss, acc=accuracy,
                                                             optimizer=Adam(), epochs=10)

# ------------------------------------------------------

# 그래프_5
# 학습 결과 시각화
def plot_metrics(train_metric, val_metric, metric_type):
  # Visualize metrics vs training Epochs
  plt.figure()
  plt.plot(range(len(train_metric)), train_metric, label = f"Training {metric_type}")
  plt.plot(range(len(val_metric)), val_metric, label = f"Validation {metric_type}")
  plt.xlabel("Epochs")
  plt.ylabel(metric_type)
  plt.legend()
  plt.title(f"{metric_type} vs Training epochs");

plot_metrics(train_losses, val_losses, "cross entropy loss")

plot_metrics(train_accs, val_accs, "accuracy")

# ------------------------------------------------------

# 모델 내보내기
# 새로운 데이터가 들어올 때 전처리 + 예측 + 후처리까지 한 번에 실행됨
class ExportModule(tf.Module):
  def __init__(self, model, preprocess, class_pred):
    # Initialize pre and postprocessing functions
    self.model = model
    self.preprocess = preprocess  # 지정된 preprocess 함수로 전처리
    self.class_pred = class_pred  # 출력 후처리 함수

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.uint8)]) 
  def __call__(self, x):
    # Run the ExportModule for new data points
    x = self.preprocess(x)
    y = self.model(x)
    y = self.class_pred(y)
    return y

# 내보내기용 전처리 & 후처리
def preprocess_test(x):   # 이게 최종적으로 사용된 전처리 함수. 0 ~ 1의 밝기처리
  # The export module takes in unprocessed and unlabeled data
  x = tf.reshape(x, shape=[-1, 784])
  x = x/255
  return x

def class_pred_test(y):   # 최종적으로 사용된 후처리 함수. 0 ~ 9 의 숫자 정하기. softmax 사용
  # Generate class predictions from MLP output
  return tf.argmax(tf.nn.softmax(y), axis=1)

mlp_model_export = ExportModule(model=mlp_model,
                                preprocess=preprocess_test,   # 전처리 함수
                                class_pred=class_pred_test)   # 후처리 함수

# SavedModel 형식으로 저장
models = tempfile.mkdtemp()
save_path = os.path.join(models, 'mlp_model_export')
tf.saved_model.save(mlp_model_export, save_path)

# 불러오기
mlp_loaded = tf.saved_model.load(save_path)

# 테스트 데이터 평가
def accuracy_score(y_pred, y):
  # Generic accuracy function
  is_equal = tf.equal(y_pred, y)
  return tf.reduce_mean(tf.cast(is_equal, tf.float32))

x_test, y_test = tfds.load("mnist", split=['test'], batch_size=-1, as_supervised=True)[0]
test_classes = mlp_loaded(x_test)
test_acc = accuracy_score(test_classes, y_test)
print(f"Test Accuracy: {test_acc:.3f}")

# 클래스별(숫자별) 정확도 확인
print("Accuracy breakdown by digit:")
print("---------------------------")
label_accs = {}
for label in range(10):
  label_ind = (y_test == label)
  # extract predictions for specific true label
  pred_label = test_classes[label_ind]
  labels = y_test[label_ind]
  # compute class-wise accuracy
  label_accs[accuracy_score(pred_label, labels).numpy()] = label
for key in sorted(label_accs):
  print(f"Digit {label_accs[key]}: {key:.3f}")

# 혼동 행렬 시각화
# 어떤 숫자를 잘 분류하고, 어떤 숫자를 헷갈려하는지 시각화. 
import sklearn.metrics as sk_metrics

def show_confusion_matrix(test_labels, test_classes):
  # Compute confusion matrix and normalize
  plt.figure(figsize=(10,10))
  confusion = sk_metrics.confusion_matrix(test_labels.numpy(), 
                                          test_classes.numpy())
  confusion_normalized = confusion / confusion.sum(axis=1, keepdims=True)
  axis_labels = range(10)
  ax = sns.heatmap(
      confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
      cmap='Blues', annot=True, fmt='.4f', square=True)
  plt.title("Confusion matrix")
  plt.ylabel("True label")
  plt.xlabel("Predicted label")

show_confusion_matrix(y_test, test_classes)