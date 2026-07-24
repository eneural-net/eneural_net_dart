# Dataset examples

Each example downloads a real public dataset (cached in the system temp dir on
first run), normalizes it, trains a network with **iRProp+**, and reports
test-set performance. All accept an optional acceleration backend argument:

```
dart run example/datasets/<name>.dart [none|auto|cpu|metal]
```

`none` (default) is pure Dart; `cpu`/`metal`/`auto` require the native libraries
(`bash native/macos/build.sh`) and fall back to pure Dart when unavailable.

| Example | Dataset (UCI) | Task | Shape |
|---|---|---|---|
| `optdigits_example.dart` | Optical Digits | 10-class classification | 64 → 32 → 10, 3823/1797 |
| `wine_quality_example.dart` | Wine Quality (red+white) | regression | 11 → 16 → 1, ~6497 rows |
| `letter_recognition_example.dart` | Letter Recognition | 26-class classification | 16 → 40 → 26, 20000 rows |

Shared helpers (download/cache, backend parsing, normalization, accuracy) live in
`common.dart`.
