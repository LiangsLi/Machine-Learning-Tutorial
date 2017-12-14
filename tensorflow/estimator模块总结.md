# Tensorflow estimator模块

## Pre-made Estimators 预定义estimator

>* `class DNNClassifier` ：DNN分类器
>* `class DNNRegressor` : DNN回归
>* `class LinearClassifier`: 线性分类器
>* `class LinearRegressor`: 线性回归
>* `class DNNLinearCombinedClassifier` : DNN与线性融合分类器
>* `class DNNLinearCombinedRegressor` : DNN与线性融合分类器

## 使用Pre-made Estimators的步骤：

1. **Write one or more dataset importing functions.** <br>For example, you might create one function to import the training set and another function to import the test set. Each dataset importing function must return two objects：<br>
	* a dictionary in which the keys are feature names and the values are Tensors (or SparseTensors) containing the corresponding feature data
	* a Tensor containing one or more labels
   input（dataset importing） functions create the TensorFlow operations that generate data for the model. 
   We can use tf.estimator.inputs.numpy_input_fn to produce the input pipeline:
	```python  
    # train_set类型：
    # Dataset(data=array(
    #     [[6.4000001, 2.79999995, 5.5999999, 2.20000005], 
    #     .......
    #     [5.5, 2.4000001, 3.70000005, 1.]], 
    #     dtype=float32), 
    #     target=array([2, .... 1, 0, 0, 1]))
    
    # Define the training inputs
  	 train_input_fn = tf.estimator.inputs.numpy_input_fn(
      	x={"x": np.array(training_set.data)},#training_set是一个命名元组
     	y=np.array(training_set.target),
     	num_epochs=None,
      	shuffle=True)
        
   # Define the test inputs
   test_input_fn = tf.estimator.inputs.numpy_input_fn(
       x={"x": np.array(test_set.data)},#test_set是一个命名元组
       y=np.array(test_set.target),
       num_epochs=1,
       shuffle=False)
	```
2. **Define the feature columns.** Each tf.feature_column identifies a feature name, its type, and any input pre-processing. <br>
只需定义特征的列，不需要定义目标值的列
	```python
    # Specify that all features have real-value data
 	feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
    ```
3. **Instantiate the relevant pre-made Estimator.**
	```python
    # Build 3 layer DNN with 10, 20, 10 units respectively.
  	classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/iris_model")
    ```
    
4. **Call a training, evaluation, or inference method.**
	```python
    # Train model.
  	classifier.train(input_fn=train_input_fn, steps=2000)
    
    # Evaluate accuracy.
  	accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    
    # Classify new flower samples.
    predictions = list(classifier.predict(input_fn=predict_input_fn))
    ```
