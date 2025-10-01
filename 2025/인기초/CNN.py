# https://www.tensorflow.org/tutorials/images/cnn

# 총평: NN보다 훨씬 쉬움. 그냥 읽으면 됨

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 데이터 부러오기: CIFAR-10 (어떻게 이름이 시팔?)
# 무려 컬러가 있는 이미지임!
# Handwriting 데이터에선 흑백이어서 한 개의 0 ~ 1 값으로 처리했잖아. 근데 이번에는 컬러라서
# 컬러 데이터를 처리하고 싶으면 0 ~ 1의 값 3개가 필요해 각각 R, G, B겠지. 
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 이건 항상 있는 학습을 위한 이미지 픽셀 값 처리(0 ~ 1)
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# 이 모델이 분류할 수 있는 클래스는 이것밖에 없음. 
# 만약 사람 사진을 넣으면 여기 있는 것들 중 하나로 분류될 것임.
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ------------------------------------------------------

# 그래프_1
# 그냥 어떤 사진들이 있는지 몇 개 뽑아서 보여줌. 학습데이터
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# ------------------------------------------------------

# 모델 정의. 그냥 보이는 그대로야. Conv2D는 NN과 CNN의 차이인데, 위치 정보를 포함하게 해 줘.
# 자세하게 알고 싶으면 https://velog.io/@byu0hyun/%EB%94%A5%EB%9F%AC%EB%8B%9D-CNN-Conv2D-Layer
# 아마 들어가서 이미지 보면 어떻게 작동하는지 알 거야. 본 적 있을걸?
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 여기서부터는 Dense, 즉 완전연결층 레이어를 사용하는데 이건 이미지 처리가 끝나고 분류를 하기 위함이야. 
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# 그 다음으로는 손실함수와 평가지표와 옵티마이저를 설정합니다. 절차에 따라
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# ------------------------------------------------------

# 그래프_2
# 학습 / 검증 정확도 시각화
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# ------------------------------------------------------

# 결과 출력
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)