# Training algorithm examples

One self-contained, runnable example per training algorithm (each trains an XOR
network). Run any with:

```
dart run example/training_algorithms/<name>_example.dart
```

## Gradient optimizers
- `backpropagation_example.dart`, `rprop_example.dart`
- `sgd_example.dart`, `sgd_nesterov_example.dart`
- `adam_example.dart`, `adamw_example.dart`, `nadam_example.dart`, `amsgrad_example.dart`
- `rmsprop_example.dart`, `adagrad_example.dart`, `adadelta_example.dart`
- `quickprop_example.dart`, `lion_example.dart`
- `resilient_propagation_example.dart` (RProp+/RProp-/iRProp+/iRProp-)

## Second-order
- `levenberg_marquardt_example.dart`, `conjugate_gradient_example.dart`, `lbfgs_example.dart`

## Population / gradient-free
- `evolution_strategy_example.dart`, `separable_cmaes_example.dart`,
  `genetic_algorithm_example.dart`, `particle_swarm_example.dart`,
  `differential_evolution_example.dart`, `simulated_annealing_example.dart`

## Techniques
- `minibatch_example.dart` — mini-batch / online training (`batchSize`)
- `dropout_example.dart` — dropout via `HiddenLayerConfig`
- `lr_schedule_example.dart` — learning-rate schedules (`lrSchedule`)

See also `example/eneural_net_optimizers_example.dart` for the name-based
registry (`trainingByName`) and JSON checkpointing.
