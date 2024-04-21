<p align="center">
  <a href="#">
    <img src="https://badges.pufler.dev/visits/stefansphtr/Train_Simple_Neural_Network" alt="Visits Badge">
    <img src="https://badges.pufler.dev/updated/stefansphtr/Train_Simple_Neural_Network" alt="Updated Badge">
    <img src="https://badges.pufler.dev/created/stefansphtr/Train_Simple_Neural_Network" alt="Created Badge">
    <img src="https://img.shields.io/github/contributors/stefansphtr/Train_Simple_Neural_Network" alt="Contributors Badge">
    <img src="https://img.shields.io/github/last-commit/stefansphtr/Train_Simple_Neural_Network" alt="Last Commit Badge">
    <img src="https://img.shields.io/github/commit-activity/m/stefansphtr/Train_Simple_Neural_Network" alt="Commit Activity Badge">
    <img src="https://img.shields.io/github/repo-size/stefansphtr/Train_Simple_Neural_Network" alt="Repo Size Badge">
    <img src="https://www.codefactor.io/repository/github/stefansphtr/Train_Simple_Neural_Network/badge" alt="CodeFactor" />
    <img src="https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?logo=tensorflow" alt="TensorFlow Badge">
    <img src="https://img.shields.io/badge/Keras-3.1.1-D00000?logo=keras" alt="Keras Badge">
  </a>
</p>

# Train Simple Neural Network

Back on April 16, 2024 I attend the lecture "Mastering Unsupervised Learning and ANN Basics: An Overview" by Bangkit Academy 2024, the lecture was held by Instructor [Rahmat Fajri](https://github.com/rfajri27). Our Instructor give us a challenge to solve the problem by building a neural network model using TensorFlow and Keras. The problem is to predict the output of the given input X and Y array. The input X and Y array are given as follows:

```python
# Defining the input array 'X'
X = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

# Defining the output array 'Y'
Y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)
```

I challenge myself to solve the problem by building a simple neural network model using TensorFlow and Keras that could solve this problem by at least 0.05% loss and could achieve this target within 10 epochs. The model architecture that I used to solve this problem is as follows:

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_12 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)             │            <span style="color: #00af00; text-decoration-color: #00af00">32</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_13 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">136</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_14 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │             <span style="color: #00af00; text-decoration-color: #00af00">9</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>
