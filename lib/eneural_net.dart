/// eNeural.net library.
library eneural_net;

export 'src/eneural_net_activation_functions.dart';
export 'src/eneural_net_ann.dart';
export 'src/eneural_net_sample.dart';
export 'src/eneural_net_scale.dart';
export 'src/eneural_net_signal.dart';
export 'src/eneural_net_tools.dart';
export 'src/eneural_net_training.dart';
export 'src/eneural_net_training_backpropagation.dart';
export 'src/eneural_net_training_rprop.dart';
export 'src/eneural_net_optimizer.dart'
    show GradientOptimizer, LearningRateScheduleBuilder;
export 'src/eneural_net_training_parameter_strategy.dart'
    show
        ParameterStrategy,
        StaticParameterStrategy,
        LearningRateScheduleStrategy,
        StepDecayStrategy,
        ExponentialDecayStrategy,
        CosineAnnealingStrategy,
        WarmupStrategy;
export 'src/optimizers/adam.dart' show Adam;
export 'src/optimizers/sgd.dart' show SGD;
export 'src/optimizers/rmsprop.dart' show RMSProp;
export 'src/optimizers/adagrad.dart' show AdaGrad;
export 'src/optimizers/adadelta.dart' show AdaDelta;
export 'src/optimizers/lion.dart' show Lion;
export 'src/optimizers/quickprop.dart' show Quickprop;
export 'src/optimizers/resilient_propagation.dart'
    show ResilientPropagation, RPropVariant;
export 'src/trainers/second_order.dart'
    show VectorTrainer, ConjugateGradient, LBFGS, LevenbergMarquardt;
export 'src/trainers/population.dart'
    show
        PopulationTrainer,
        EvolutionStrategy,
        SeparableCMAES,
        GeneticAlgorithm,
        ParticleSwarm,
        DifferentialEvolution,
        SimulatedAnnealing;
export 'src/eneural_net_training_registry.dart'
    show
        TrainingD,
        AnnD,
        SamplesD,
        TrainingBuilderFn,
        registerTraining,
        registeredTrainings,
        trainingByName,
        saveTrainingCheckpoint,
        restoreTrainingCheckpoint;
export 'src/native/native_backend.dart' show NativeBackend;
export 'src/native/native_trainer.dart' show NativeBackpropagation, NativeRProp;
export 'src/native/webgpu/webgpu_trainer.dart'
    show WebGpuBackpropagation, WebGpuRProp, WebGpuTrainerMixin;
