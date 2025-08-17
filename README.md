# Advanced Deep Learning: Optimized Custom Training Loop with TensorFlow

This project demonstrates a robust understanding of low-level TensorFlow by implementing a high-accuracy digit classifier from scratch. By moving beyond high-level APIs like `model.fit()`, it showcases the ability to build, optimize, and control the entire deep learning training pipeline using `tf.GradientTape`.

## Problem Statement and Goal of Project

The primary goal was to implement a high-performance image classifier for the MNIST dataset while maintaining full control over the training process. This project was designed to showcase proficiency in advanced deep learning techniques, including:

  - Building an efficient, end-to-end data pipeline with `tf.data`.
  - Implementing a custom training and validation loop using `tf.GradientTape`.
  - Integrating key regularization and optimization techniques like **Batch Normalization**, **Dropout**, and **dynamic learning rate scheduling** to maximize model accuracy.
  - Conducting a thorough final evaluation, including a confusion matrix, to analyze model performance in detail.

## Solution Approach

The solution is a complete, custom-built training pipeline that prioritizes both efficiency and performance.

  - **Data Preparation**: The MNIST dataset was loaded using `keras.datasets` and split into training, validation, and test sets. Input images were flattened and normalized to a `[0, 1]` range. For efficiency, labels were kept as integers and paired with a `SparseCategoricalCrossentropy` loss function.

  - **Efficient Data Pipelines**: `tf.data.Dataset` was used to create highly efficient data loaders. The training pipeline includes shuffling, batching, and prefetching (`.prefetch(tf.data.AUTOTUNE)`) to ensure the GPU is always saturated with data, minimizing I/O bottlenecks.

  - **Model Architecture**: An MLP (Multi-Layer Perceptron) was built using the Keras Functional API. To enhance performance and combat overfitting, the architecture incorporates:

      - **Batch Normalization** layers to stabilize and accelerate training.
      - **Dropout** layers to provide regularization and improve generalization.
      - A final `Dense` layer without activation (logits), as the `from_logits=True` argument is handled directly in the loss function for better numerical stability.

  - **Custom Training Loop**: The core of the project is a custom training loop built with `tf.GradientTape`. This loop gives full control over:

      - The forward pass and loss calculation.
      - Gradient computation and application to model weights.
      - Manual implementation of **Early Stopping** and **ReduceLROnPlateau** logic to dynamically adjust the learning rate and prevent unnecessary training epochs.

  - **Evaluation**: After training, the model's final performance was validated on the unseen test dataset. A **confusion matrix** was generated to provide a clear, class-by-class visualization of the model's predictive accuracy.

## Technologies & Libraries

  - **Primary Framework**: TensorFlow 2.10
  - **Core Libraries**: Keras, NumPy
  - **Data Visualization**: Matplotlib
  - **Metrics & Analysis**: scikit-learn

## Description about Dataset

This project utilizes the **MNIST** dataset, a classic benchmark in computer vision. It contains 70,000 grayscale images of handwritten digits (0 through 9), each of size `28x28` pixels. The dataset is pre-divided into 60,000 training images and 10,000 testing images. For this project, a 10% validation set was further carved out from the training data.

## Installation & Execution Guide

To replicate this project, please follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/imehranasgari/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    *(A `requirements.txt` file containing `tensorflow`, `numpy`, `matplotlib`, and `scikit-learn` should be included in the repository.)*

3.  **Run the Notebook:**
    Launch Jupyter and open the `custom_model.ipynb` notebook to execute the cells sequentially.

    ```bash
    jupyter notebook
    ```

## Key Results / Performance

The primary outcome is a highly accurate digit classifier trained with a custom, low-level loop. This approach not only achieves excellent results but also demonstrates the ability to implement and manage complex training logic manually.

  - The model successfully converges, with validation accuracy consistently improving over epochs.
  - The final evaluation on the test set confirms the model's high generalization performance.
  - The loss and accuracy curves illustrate stable training, and the confusion matrix provides a clear visualization of the classifier's strengths and weaknesses.

## Screenshots / Sample Output

*This file was intentionally created to demonstrate skills in implementing and explaining machine learning models, rather than solely focusing on achieving the highest evaluation metrics. The simple approach is for learning, benchmarking, and illustrating fundamental concepts.*

**Training & Validation Performance Curves**

**Test Set Confusion Matrix**

## Additional Learnings / Reflections

This project reinforced my understanding of what happens "under the hood" of high-level frameworks. By manually implementing the training loop, I gained direct control over gradient updates and was able to integrate custom logic for learning rate reduction and early stopping without relying on standard Keras callbacks. This approach is invaluable for non-standard research or when fine-tuning model behavior is critical. It proves that a deep understanding of the fundamentals can lead to more efficient and powerful models, even with a relatively simple architecture.

-----

## 👤 Author

**Mehran Asgari**

  - **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)
  - **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari)

-----

## 📄 License

This project is licensed under the Apache 2.0 License – see the `LICENSE` file for details.

-----

> 💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*