# Tensorflow estimator模块

## Pre-made Estimators 预定义estimator

>* `class DNNClassifier` ：DNN分类器
>* `class DNNRegressor` : DNN回归
>* `class LinearClassifier`: 线性分类器
>* `class LinearRegressor`: 线性回归
>* `class DNNLinearCombinedClassifier` : DNN与线性融合分类器
>* `class DNNLinearCombinedRegressor` : DNN与线性融合分类器

## 使用Pre-made Estimators的步骤：

1. **Write one or more dataset importing functions.构建dataset输入函数** <br>For example, you might create one function to import the training set and another function to import the test set. Each dataset importing function must return two objects（该函数应当返回一下两个值）：<br>
	* a dictionary in which the keys are feature names and the values are Tensors (or SparseTensors) containing the corresponding feature data（这个值应当是一个字典，其key值为特征的名称，value是特征的张量值）
	* a Tensor containing one or more labels（一个包含所有label的张量）
	
 	input（dataset importing） functions create the TensorFlow operations that generate data for the model. We can use 	  tf.estimator.inputs.numpy_input_fn to produce the input pipeline:
   
```python  
    # train_set类型如下：
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
只需定义特征的列，不需要定义目标值的列，如果有多个特征，应当全部定义出来
	```python
    # Specify that all features have real-value data
 	feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
    ```
3. **Instantiate the relevant pre-made Estimator.生成预定义估计器实例**
	```python
    # Build 3 layer DNN with 10, 20, 10 units respectively.
  	classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/iris_model")
    ```
    
4. **Call a training, evaluation, or inference method.调用训练、评估、预测方法**
	```python
    # Train model.
  	classifier.train(input_fn=train_input_fn, steps=2000)
    
    # Evaluate accuracy.
  	accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    
    # Classify new flower samples.
    predictions = list(classifier.predict(input_fn=predict_input_fn))
    ```

## 使用预定义DNN分类器

### tf.estimator.DNNClassifier()函数解释：
参数：
* hidden_units: （隐层单元数，所有层都是全连接的）Ex. `[64, 32]` means first layer has 64 nodes and second one has 32.
* feature_columns:（特征列列表，这里需要指明特征列，应当是`_FeatureColumn`子类的实例）
* model_dir:（模型参数等的保存路径）This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
* n_classes:（最后输出的分类数目，默认为2，至少大于1）
* weight_column:（权重列？？） A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
        then weight_column.normalizer_fn is applied on it to get weight tensor.
* label_vocabulary:（指明string类型的label对应的数值，如果已经将label转换为整数或者浮点数的话，则不需要指明这个值） A list of strings represents possible label values. If
        given, labels must be string type and have any value in
        `label_vocabulary`. If it is not given, that means labels are
        already encoded as integer or float within [0, 1] for `n_classes=2` and
        encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .
        Also there will be errors if vocabulary is not provided and labels are
        string.
* optimizer:（优化器，默认是adagrad） An instance of `tf.Optimizer` used to train the model. Defaults
        to Adagrad optimizer.
* activation_fn: （激活函数，默认是relu）Activation function applied to each layer. If `None`, will
        use `tf.nn.relu`.
* dropout: （是否使用dropout）When not `None`, the probability we will drop out a given
        coordinate.
* input_layer_partitioner: Optional. Partitioner for input layer. Defaults
        to `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
* config: `RunConfig` object to configure the runtime settings.
