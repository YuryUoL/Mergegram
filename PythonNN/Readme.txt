KerasConfiguration.py contains layers from Perslay-paper in Tensorflow v2 format

PerslayTools.py contains preprocessing methods that helps transforming given data to data that is accepted by this module.

Example of usage:

(1) Define layers: / Parameters / 
 - Coefficent layer 
 - Functional layer
 - Permutation invariant layer


model = SinglePerslayModel(CoefficentLDDL(),
                                             SequencePeL([PeL(dimension=50), PeL(dimension=50, operationalLayer=PermutationMaxLayer)]),
                                          TopK(50),10)

(2) Define optimzers, loss function , other stuff required for neural network ....

model.compile(optimizer=optimizer,
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])

(3) Run learning phase

model.fit(train_diags[0], train_labels, epochs=num_epochs, batch_size=128)
                test_loss_ok, test_acc_ok = model.evaluate(test_diags[0], test_labels, verbose=2)

(4) Make final predictions 

  test_loss_ok, test_acc_ok = model.evaluate(test_diags[0], test_labels, verbose=2)

