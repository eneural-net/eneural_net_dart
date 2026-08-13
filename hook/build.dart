// Dart build hook for the `eneural_net` native accelerators.
//
// This hook is run automatically by the Dart/Flutter SDK (`dart run`,
// `dart test`, `dart build`, `flutter build`, ...). It compiles the native
// backend under `native/` for the target platform and publishes each resulting
// dynamic library as a `CodeAsset`, so the `@Native` bindings in
// `lib/src/native/native_accelerator_io.dart` resolve without any manual build
// step.
//
//   macOS          -> libeneural_cpu_<arch>.dylib   (Accelerate, via swiftc)
//                     libeneural_metal_<arch>.dylib (Metal + MPS, via swiftc)
//   Linux/Windows  -> libeneural_cuda_<arch>.{so,dll} (cuBLAS, via nvcc)
//
// The compiler invocations mirror the manual build scripts under `native/`
// (`macos/cpu/build.sh`, `macos/metal/build.sh`, `cuda/build.sh`,
// `cuda/build.bat`). Those scripts are still supported for local/debug builds,
// but this hook is the primary build path — it cannot call them, since they
// write into the package directory, which is read-only in the pub cache.
//
// This hook is deliberately total: it never throws and never fails the build.
// The SDK runs it for *every* `dart run`/`dart test`, including web platforms
// and machines without any native toolchain. When a backend cannot be built the
// hook simply emits no asset for it, and the Dart side falls back to the
// pure-Dart training path.
import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    try {
      await _build(input, output);
    } catch (e, s) {
      stderr.writeln(
        'eneural_net: build hook failed, skipping native '
        'acceleration: $e\n$s',
      );
    }
  });
}

Future<void> _build(BuildInput input, BuildOutputBuilder output) async {
  if (!input.config.buildCodeAssets) return;

  final code = input.config.code;

  // Every backend is a dynamic library. When the embedder demands static
  // linking (Flutter iOS) there is nothing this hook can provide, and emitting
  // a `DynamicLoadingBundled` asset would fail output validation.
  if (code.linkModePreference == LinkModePreference.static) return;

  final packageRoot = input.packageRoot;

  // Declared unconditionally, including on runs that build nothing: a hook run
  // that failed for a missing toolchain must still be re-run when the native
  // sources change.
  output.dependencies.addAll(
    _nativeSources(packageRoot).where((uri) => File.fromUri(uri).existsSync()),
  );

  final targetOS = code.targetOS;
  final targetArch = code.targetArchitecture;

  // The library file names carry the architecture, using the toolchain spelling
  // (`x86_64`) rather than Dart's (`x64`).
  final archSuffix = _archSuffix(targetArch);
  if (archSuffix == null) return;

  final outDir = Directory.fromUri(
    input.outputDirectoryShared.resolve('${targetOS.name}_${targetArch.name}/'),
  );
  outDir.createSync(recursive: true);

  if (targetOS == OS.macOS) {
    await _buildMacOS(input, output, outDir.uri, archSuffix);
  } else if (targetOS == OS.linux || targetOS == OS.windows) {
    await _buildCuda(input, output, outDir.uri, archSuffix, targetOS);
  }
  // Other targets (iOS, Android, Fuchsia) have no native backend yet.
}

// ============================================================
// macOS: Accelerate (CPU) + Metal (GPU), both built with swiftc
// ============================================================

Future<void> _buildMacOS(
  BuildInput input,
  BuildOutputBuilder output,
  Uri outDir,
  String archSuffix,
) async {
  final swiftc = await _which('swiftc');
  if (swiftc == null) {
    stderr.writeln(
      'eneural_net: `swiftc` not found on PATH, skipping the '
      'macOS CPU/Metal backends (install the Xcode command line tools to '
      'enable native acceleration).',
    );
    return;
  }

  // Match the deployment target the SDK asked for, so the produced dylib is not
  // newer than the application that bundles it. `swiftc` cross-compiles between
  // arm64 and x86_64 fine, so the host architecture is irrelevant here.
  final targetVersion = _macOSTargetVersion(input.config.code);
  final targetTriple = targetVersion == null
      ? null
      : '$archSuffix-apple-macosx$targetVersion';

  final cpuDir = input.packageRoot.resolve('native/macos/cpu/');
  final cpuLib = outDir.resolve('libeneural_cpu_$archSuffix.dylib');
  if (await _run(
    swiftc,
    [
      '-O',
      '-whole-module-optimization',
      '-emit-library',
      if (targetTriple != null) ...['-target', targetTriple],
      cpuDir.resolve('EnnCpuBackend.swift').toFilePath(),
      '-o',
      cpuLib.toFilePath(),
      '-framework',
      'Accelerate',
      '-framework',
      'Foundation',
    ],
    workingDirectory: outDir,
    expectedOutput: cpuLib,
    what: 'macOS CPU (Accelerate) backend',
  )) {
    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'eneural_cpu',
        linkMode: DynamicLoadingBundled(),
        file: cpuLib,
      ),
    );
  }

  final metalDir = input.packageRoot.resolve('native/macos/metal/');
  final metalLib = outDir.resolve('libeneural_metal_$archSuffix.dylib');
  if (await _run(
    swiftc,
    [
      '-O',
      '-whole-module-optimization',
      '-emit-library',
      if (targetTriple != null) ...['-target', targetTriple],
      metalDir.resolve('EnnMetalShaders.swift').toFilePath(),
      metalDir.resolve('EnnMetalBackend.swift').toFilePath(),
      '-o',
      metalLib.toFilePath(),
      '-framework',
      'Foundation',
      '-framework',
      'Metal',
      '-framework',
      'MetalPerformanceShaders',
    ],
    workingDirectory: outDir,
    expectedOutput: metalLib,
    what: 'macOS Metal (GPU) backend',
  )) {
    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'eneural_metal',
        linkMode: DynamicLoadingBundled(),
        file: metalLib,
      ),
    );
  }
}

/// The macOS deployment target requested by the SDK, or `null` when the
/// embedder did not provide a macOS configuration.
int? _macOSTargetVersion(CodeConfig code) {
  try {
    return code.macOS.targetVersion;
  } catch (_) {
    return null;
  }
}

// ============================================================
// Linux/Windows: CUDA (cuBLAS), built with nvcc
// ============================================================

/// The SM targets built into the fat binary, mirroring `native/cuda/build.sh`.
const _cudaArchs = [70, 75, 80, 86, 89, 90];

Future<void> _buildCuda(
  BuildInput input,
  BuildOutputBuilder output,
  Uri outDir,
  String archSuffix,
  OS targetOS,
) async {
  // `nvcc` has no usable cross-compilation story here: it drives the host C++
  // toolchain. Only build when the target is the machine we are running on.
  if (targetOS != OS.current ||
      input.config.code.targetArchitecture != Architecture.current) {
    return;
  }

  // Silent skip: most Linux/Windows machines have no CUDA toolkit, and this
  // hook runs on every `dart run`/`dart test`, including web test runs.
  final nvcc = await _which('nvcc');
  if (nvcc == null) return;

  final cudaDir = input.packageRoot.resolve('native/cuda/');
  final ext = targetOS == OS.windows ? 'dll' : 'so';
  final cudaLib = outDir.resolve('libeneural_cuda_$archSuffix.$ext');

  if (await _run(
    nvcc,
    [
      '-O3',
      '--shared',
      // Windows builds a DLL with the MSVC toolchain, which has no -fPIC.
      if (targetOS != OS.windows) ...['--compiler-options', '-fPIC'],
      '--use_fast_math',
      '-lcublas',
      for (final arch in _cudaArchs)
        '-gencode=arch=compute_$arch,code=sm_$arch',
      '-I',
      cudaDir.toFilePath(),
      '-o',
      cudaLib.toFilePath(),
      cudaDir.resolve('EnnCudaBackend.cu').toFilePath(),
    ],
    workingDirectory: outDir,
    expectedOutput: cudaLib,
    what: 'CUDA backend',
  )) {
    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'eneural_cuda',
        linkMode: DynamicLoadingBundled(),
        file: cudaLib,
      ),
    );
  }
}

// ============================================================
// Process helpers — none of these may throw.
// ============================================================

/// Resolves [executable] on `PATH`, or `null` when it is not installed.
Future<String?> _which(String executable) async {
  try {
    final result = await Process.run(Platform.isWindows ? 'where' : 'which', [
      executable,
    ]);
    if (result.exitCode != 0) return null;
    final lines = (result.stdout as String)
        .split('\n')
        .map((l) => l.trim())
        .where((l) => l.isNotEmpty);
    return lines.isEmpty ? null : lines.first;
  } catch (_) {
    return null;
  }
}

/// Runs [executable], returning whether it succeeded *and* produced
/// [expectedOutput].
///
/// Reports failures on stderr and returns `false` rather than throwing: a
/// missing or broken toolchain must degrade to "no native acceleration", not
/// break the whole build.
Future<bool> _run(
  String executable,
  List<String> arguments, {
  required Uri workingDirectory,
  required Uri expectedOutput,
  required String what,
}) async {
  try {
    final result = await Process.run(
      executable,
      arguments,
      workingDirectory: workingDirectory.toFilePath(),
    );

    if (result.exitCode != 0) {
      stderr.writeln(
        'eneural_net: failed to build the $what '
        '(exit code ${result.exitCode}):',
      );
      stderr.writeln('  \$ $executable ${arguments.join(' ')}');
      stderr.writeln(result.stdout);
      stderr.writeln(result.stderr);
      return false;
    }

    if (!File.fromUri(expectedOutput).existsSync()) {
      stderr.writeln(
        'eneural_net: the $what build reported success but did '
        'not produce ${expectedOutput.toFilePath()}.',
      );
      return false;
    }

    return true;
  } catch (e) {
    stderr.writeln('eneural_net: could not run $executable for the $what: $e');
    return false;
  }
}

// ============================================================
// Shared bits
// ============================================================

/// Dart's [Architecture] names mapped to the toolchain spelling used in the
/// library file names. `null` for architectures with no native backend.
String? _archSuffix(Architecture arch) => switch (arch) {
  Architecture.arm64 => 'arm64',
  Architecture.x64 => 'x86_64',
  _ => null,
};

/// The non-Dart sources this hook compiles. Registered as hook dependencies so
/// that editing a backend re-triggers the build.
List<Uri> _nativeSources(Uri packageRoot) => [
  packageRoot.resolve('native/macos/cpu/EnnCpuBackend.swift'),
  packageRoot.resolve('native/macos/metal/EnnMetalShaders.swift'),
  packageRoot.resolve('native/macos/metal/EnnMetalBackend.swift'),
  packageRoot.resolve('native/cuda/EnnCudaBackend.cu'),
  packageRoot.resolve('native/cuda/cuda_common.h'),
];
