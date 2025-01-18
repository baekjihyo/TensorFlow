### 1. Import Libraries
tensorflow, matplotlib, numpy, pandas etc

---

### 2. Process Data
(usually using pandas)

---

### 3. Build a Model

**a. Make Layers:**
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(numbers of node, activation = 'activation_fn')
])
```

**b. compile model:**
```
model.compile(optimizer = 'optimizer_fn', loss = 'loss_fn', metrics = ['metrics'])
```

---

### 4. Train the Model

```
model.fit(xData, yData, epochs = num)
```

---

### 5. Predict

```
model.predict( [a, b, c ...] )
```