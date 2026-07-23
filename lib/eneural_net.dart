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
export 'src/native/native_backend.dart' show NativeBackend;
export 'src/native/native_trainer.dart' show NativeBackpropagation, NativeRProp;
export 'src/native/webgpu/webgpu_trainer.dart'
    show WebGpuBackpropagation, WebGpuRProp, WebGpuTrainerMixin;
